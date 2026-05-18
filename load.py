import torch
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print("device = " + str(device))
torch.set_printoptions(sci_mode=False)

df = pd.read_csv('data/standardData.csv')
treatment = torch.tensor(df.iloc[:,1]).float().unsqueeze(1)
outcome = torch.tensor(df.iloc[:,2]).float().unsqueeze(1)
covars = torch.tensor(df.iloc[:,3:].to_numpy()).float()
obs_count = covars.shape[0]

