import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import IST_Proxy_Dataset
from torch.utils.data import DataLoader
from cevae_continuous import cevae_continuous
import wandb
from tqdm import tqdm
import sys
sys.path.append('.')

def main():
    device = "cpu"
    ihdp = pd.read_csv('data/ihdp_project2.csv.gz', index_col=0)
    ihdp.drop(columns=['Y_cf', 'Y1', 'Y0', 'ITE', 'ps'], inplace=True)

    # Define variables
    x_cat_variables = ['X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15',
                       'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X24', 'X25']
    x_con_variables = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
    x_variables = x_cat_variables + x_con_variables
    t_variable = 'T'
    y_variable = 'Y'
    cate_variable = 'CATE'
    print("True ATE: ", ihdp[cate_variable].mean())

    # Split the dataset into train and test sets
    train_data, val_data = train_test_split(ihdp, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = IST_Proxy_Dataset(train_data, x_variables, t_variable, y_variable)
    val_dataset = IST_Proxy_Dataset(val_data, x_variables, t_variable, y_variable)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=15253, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=15253, shuffle=False)

    # Define the network
    model = cevae_continuous(len(x_variables), len(x_con_variables)).to(device)
    max_epochs = 2000

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    wandb.init(project="CPH200B A3", entity="zhongyuan_liang", name="CEVAE ihdp", sync_tensorboard=True)
    wandb.watch(model)
    for __ in tqdm(range(max_epochs)):
        for __, sample in enumerate(train_loader):
            sample = {key: value.to(device) for key, value in sample.items()}
            sample['t'] = sample['t'].unsqueeze(-1)
            sample['y'] = sample['y'].unsqueeze(-1)
            model.train()
            optimizer.zero_grad()
            output = model(sample)

            print('pred t mean: ', output[0].mean())
            print('t_encoder weights mean: ', model.t_encoder[0].weight.mean())
            print('t_encoder bias mean: ', model.t_encoder[0].bias.mean())
            print('t_encoder weights grad mean: ', model.t_encoder[0].weight.grad.mean())
            print('t_encoder bias grad mean: ', model.t_encoder[0].bias.grad.mean())
            
            loss_with_sampled_t, q_t_loss, q_y_loss, z_kl_loss, p_t_loss, p_x_con_loss, p_x_dis_loss, p_y_loss = model.train_loss(output, sample)
            loss_with_sampled_t.backward()
            optimizer.step()

            wandb.log({"Train Overall Loss": loss_with_sampled_t})
            wandb.log({"Train Encoder t Reconstruction Loss": q_t_loss})
            wandb.log({"Train Encoder y Reconstruction Loss": q_y_loss})
            wandb.log({"Train Z KL Loss": z_kl_loss})
            wandb.log({"Train Decoder t Reconstruction Loss": p_t_loss})
            wandb.log({"Train Decoder x Continuous Reconstruction Loss": p_x_con_loss})
            wandb.log({"Train Decoder x Discrete Reconstruction Loss": p_x_dis_loss})
            wandb.log({"Train Decoder y Reconstruction Loss": p_y_loss})

        for __, sample in enumerate(val_loader):
            sample = {key: value.to(device) for key, value in sample.items()}
            sample['t'] = sample['t'].unsqueeze(-1)
            sample['y'] = sample['y'].unsqueeze(-1)
            model.eval()
            output = model(sample)
            loss_with_sampled_t = model.train_loss(output, sample)
            output = model.inference(sample)
            loss_with_true_t = model.inference_loss(output, sample)
            cate = model.cate(sample)

            wandb.log({"Validation Loss with Sampled t": loss_with_sampled_t})
            wandb.log({"Validation Loss with True t": loss_with_true_t})
            wandb.log({"ATE": cate.mean()})
            wandb.log({"CATE MSE": nn.MSELoss()(cate, torch.zeros_like(cate))})
    wandb.finish()

if __name__ == "__main__":
    main()