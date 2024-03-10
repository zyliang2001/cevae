from torch.utils.data import Dataset, DataLoader
import torch
  
class IST_Proxy_Dataset(Dataset):
    def __init__(self, dataframe, x_variables, t_variable, y_variable):
        self.data_frame = dataframe
        self.x_proxy = self.data_frame[x_variables].values
        self.treatment = self.data_frame[t_variable].values
        self.outcome = self.data_frame[y_variable].values
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        sample = {
            'x': torch.tensor(self.x_proxy[index], dtype=torch.float32),
            't': torch.tensor(self.treatment[index], dtype=torch.float32),
            'y': torch.tensor(self.outcome[index], dtype=torch.float32)}
        return sample