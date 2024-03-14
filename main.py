import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import IST_Proxy_Dataset
from torch.utils.data import DataLoader
from cevae_pro_max import cevae_pro_max
import wandb
from tqdm import tqdm
import sys
sys.path.append('.')

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    IST_proxy_data = pd.read_csv('data/IST_Done.csv')

    # Define variables
    x_cat_variables = ['STYPE_PROXY_LACS', 'STYPE_PROXY_PACS', 'STYPE_PROXY_POCS', 'STYPE_PROXY_TACS', 'STYPE_PROXY_OTH',
            'RDEF4_PROXY_Y' ,'RDEF4_PROXY_N', 'RDEF4_PROXY_C', 'RDEF5_PROXY_Y', 'RDEF5_PROXY_N', 'RDEF5_PROXY_C',
            'RDEF6_PROXY_Y', 'RDEF6_PROXY_N', 'RDEF6_PROXY_C', 'RCONSC_PROXY_D', 'RCONSC_PROXY_F', 'RCONSC_PROXY_U',
            'RATRIAL_PROXY_N', 'RATRIAL_PROXY_Y']
    x_con_variables = ['AGE_PROXY', 'RDEF_PROXY', 'RHEP24_RASP3_PROXY']
    x_variables = x_cat_variables + x_con_variables
    t_variable = 'RXASP'
    y_variable = 'ID14'

    # Split the dataset into train and test sets
    train_data, temp = train_test_split(IST_proxy_data, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp, test_size=0.5, random_state=42)

    # Create datasets
    train_dataset = IST_Proxy_Dataset(train_data, x_variables, t_variable, y_variable)
    val_dataset = IST_Proxy_Dataset(val_data, x_variables, t_variable, y_variable)
    test_dataset = IST_Proxy_Dataset(test_data, x_variables, t_variable, y_variable)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=15253, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=15253, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=15253, shuffle=False)

    # Define the network
    model = cevae_pro_max(len(x_variables), len(x_con_variables)).to(device)
    max_epochs = 100

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    wandb.init(project="CPH200B A3", entity="zhongyuan_liang", name="cevae_pro_max_balacing", sync_tensorboard=True)
    wandb.watch(model)
    for __ in tqdm(range(max_epochs)):
        for __, sample in enumerate(train_loader):
            sample = {key: value.to(device) for key, value in sample.items()}
            model.train()
            optimizer.zero_grad()
            output = model(sample)
            loss_with_true_t, q_t_loss, q_y_loss, z_kl_loss, mmd_loss, p_t_loss, p_x_con_loss, p_x_dis_loss, p_y_loss = model.train_loss(output, sample)
            loss_with_true_t.backward()
            optimizer.step()
            cate = model.cate(sample)

            wandb.log({"Train Overall Loss": loss_with_true_t})
            wandb.log({"Train Encoder t Reconstruction Loss": q_t_loss})
            wandb.log({"Train Encoder y Reconstruction Loss": q_y_loss})
            wandb.log({"Train Z KL Loss": z_kl_loss})
            wandb.log({"Train Z MMD Loss": mmd_loss})
            wandb.log({"Train Decoder t Reconstruction Loss": p_t_loss})
            wandb.log({"Train Decoder x Continuous Reconstruction Loss": p_x_con_loss})
            wandb.log({"Train Decoder x Discrete Reconstruction Loss": p_x_dis_loss})
            wandb.log({"Train Decoder y Reconstruction Loss": p_y_loss})
            wandb.log({"Train_ATE": cate.mean()})

        for __, sample in enumerate(val_loader):
            sample = {key: value.to(device) for key, value in sample.items()}
            model.eval()
            output = model(sample)
            loss_with_true_t = model.train_loss(output, sample)[0]
            cate = model.cate(sample)

            wandb.log({"Validation Loss with True t": loss_with_true_t})
            wandb.log({"Validation_ATE": cate.mean()})

        for __, sample in enumerate(test_loader):
            sample = {key: value.to(device) for key, value in sample.items()}
            model.eval()
            output = model(sample)
            loss_with_true_t = model.train_loss(output, sample)[0]
            cate = model.cate(sample)

            wandb.log({"Test Loss with True t": loss_with_true_t})
            wandb.log({"Test_ATE": cate.mean()})
    wandb.finish()

if __name__ == "__main__":
    main()