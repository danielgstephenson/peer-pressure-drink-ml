import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_printoptions(sci_mode=False)

df = pd.read_csv('data/cleanData.csv')
df.columns

numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
numeric_df = df[numeric_cols]
numeric_tensor = torch.tensor(numeric_df.to_numpy()).to(torch.float32)
numeric_mean = torch.mean(numeric_tensor,dim=0).unsqueeze(0)
numeric_sd = torch.std(numeric_tensor,dim=0).unsqueeze(0)
numeric_normal = (numeric_tensor - numeric_mean) / numeric_sd

category_cols = [col for col in df.columns if not is_numeric_dtype(df[col])]
category_df = df[category_cols].astype('category')
one_hot_tensors: list[Tensor] = []

for col in category_df.columns:
    codes = category_df[col].cat.codes
    code_tensor = torch.tensor(codes,dtype=torch.long)
    one_hot = F.one_hot(code_tensor).to(torch.float32)
    one_hot_tensors.append(one_hot)

target = numeric_normal[:,0].unsqueeze(1).to(device)
treatment = one_hot_tensors[0].to(device)

numeric_covars = numeric_normal[:,1:]
one_hot_covars = torch.cat(one_hot_tensors[1:],dim=1)
covars = torch.cat([numeric_covars,one_hot_covars],dim=1).to(device)
