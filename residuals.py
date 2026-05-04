import torch
from torch import Tensor
from loader import outcome, treatment_one_hot, row_count
import torch.nn.functional as F
import pandas as pd

print('Construct Residuals')

outcome_df = pd.read_csv('output/outcome.csv')
outcome_output = torch.tensor(outcome_df['output'].to_numpy())
outcome_residual = outcome.squeeze(1) - outcome_output

treatment_df = pd.read_csv('output/treatment.csv')
treatment_output = torch.tensor(treatment_df[['prob0','prob1','prob2','prob3']].to_numpy())
treatment_residual = treatment_one_hot - treatment_output

outcome_file = open('output/residuals.csv', mode='w', buffering=1)
_ = outcome_file.write('outcome,treatment1,treatment2,treatment3\n')
for row in range(row_count):
    s = ''
    s += f'{outcome_residual[row]},'
    s += f'{treatment_residual[row,1]},{treatment_residual[row,2]},{treatment_residual[row,3]}'
    _ = outcome_file.write(f'{s}\n')

print('Residuals Saved')