import sys
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

from loader import target, treatment, covars, treatment_names, treatment_codes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_printoptions(sci_mode=False)

# DONE:
# construct an accurate version of the groupSize variable
# normalize the numeric variables
# setup the structure of the model
# setup optimization
# setup training and testing data randomization
# produce out of sample predictions
# identify the early stopping time (Around 40)
# measure treatment effects using counterfactal predictions
# estimate the variance of the predictions (i.e. measurement error)

# TO DO:
# permutation test to identify significance
# increase the number of trials (to reduce measurement error)
# consider which variables to include (include only relevant variables)

input_size = treatment.shape[1] + covars.shape[1]
hidden_layer_count = 4
hidden_layer_size = 100

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = F.relu
        self.init_layer = nn.Linear(input_size, hidden_layer_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layer_count):
            self.hidden_layers.append(nn.Linear(hidden_layer_size,hidden_layer_size))
        self.final_layer = nn.Linear(hidden_layer_size, 1)
    def forward(self, treatment: Tensor, covars: Tensor) -> Tensor:
        x: Tensor = torch.cat([treatment, covars], dim=1)
        x = self.activation(self.init_layer(x))
        for i in range(hidden_layer_count):
            h = self.hidden_layers[i]
            x = self.activation(h(x))
        x = self.final_layer(x)
        return x
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)

dataset = TensorDataset(target, treatment, covars)

# Increase the step count from 50 to 100 if stopping times are picked separately for each trial.
def get_output_path(dataset: TensorDataset, observation: int, trial_count = 10, step_count= 50):
    test_dataset = Subset(dataset,[observation])
    train_dataset = Subset(dataset,[i for i in range(len(dataset)) if i != observation])
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=True
    )
    test_loss_path = torch.zeros(step_count, trial_count).to(device)
    output_path = torch.zeros(step_count, trial_count, 4).to(device)
    treatments = torch.eye(4).to(device)
    for trial in range(trial_count):
        model = Model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        for step in range(step_count):
            train_batch: tuple[Tensor, Tensor, Tensor] = next(iter(train_dataloader))
            train_target, train_treatment, train_covars = train_batch
            train_output = model(train_treatment,train_covars)
            train_loss = F.mse_loss(train_output, train_target)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                test_batch: tuple[Tensor, Tensor, Tensor] = next(iter(test_dataloader))
                test_target, test_treatment, test_covars = test_batch
                test_output = model(test_treatment, test_covars) 
                test_loss = F.mse_loss(test_output, test_target)
                test_loss_value = test_loss.detach().cpu().item()
                test_loss_path[step, trial] = test_loss_value
                repeat_covars = test_covars.repeat([4,1])
                outputs = model(treatments, repeat_covars)
                output_path[step, trial, :] = outputs[:,0]
    return output_path, test_loss_path

def get_permute_dataset() -> TensorDataset:
    shuffled_treatment = treatment.clone()
    original_indices = np.where((treatment_codes == 2) | (treatment_codes == 3))[0]
    shuffled_indices = np.random.permutation(original_indices)
    shuffled_treatment[original_indices,:] = shuffled_treatment[shuffled_indices,:]
    shuffled_dataset = TensorDataset(target, shuffled_treatment, covars)
    return shuffled_dataset

def get_predictions(dataset: TensorDataset, step_count = 40, trial_count = 20) -> tuple[Tensor, ...]:
    test_loss_tensor = torch.zeros(step_count, trial_count, len(dataset)).to(device)
    output_tensor = torch.zeros(step_count, trial_count, len(dataset), 4).to(device)
    for observation in range(len(dataset)):
        output_path, test_loss_path = get_output_path(dataset, observation,trial_count,step_count)
        test_loss_tensor[:,:,observation] = test_loss_path
        output_tensor[:,:,observation,:] = output_path
        if observation % 2 == 0: print('.',end='',flush=True)
    final_predictions = output_tensor[-1,:,:,:]
    final_test_loss = test_loss_tensor[-1,:,:]
    print('\n')
    return final_predictions, final_test_loss

step_count = 40
permutation_count = 100
trial_count = 10

# Construct a modified dataset where the treatment variable is uniformly set to zero.
# Collect multiple estimates of the test loss under each data set. (modified vs. original)
# Pick the optimal stopping time separately for each trial. (to ensure trials are independent)
# Use a Mann-Whitney test to compare the test loss distributions between the two datasets.

original_dataset = TensorDataset(target, treatment, covars)
original_predictions, original_test_loss = get_predictions(original_dataset, step_count, trial_count)
original_mean_test_loss = torch.mean(original_test_loss).item()
original_test_loss_file = open('output/original_test_loss.csv', mode='a', buffering=1)
original_test_loss_file.write(f'{original_mean_test_loss}\n')

permute_test_loss_file = open('output/permute_test_loss.csv', mode='a', buffering=1) 
print(f'Begin Permutation Test')
for i in range(permutation_count):
    permute_dataset = get_permute_dataset()
    permute_predictions, permute_test_loss = get_predictions(permute_dataset, step_count, trial_count)
    mean_permute_test_loss = torch.mean(permute_test_loss).item()
    permute_test_loss_file.write(f'{mean_permute_test_loss}\n')
    print(f'Permutation Trial {i+1}: {mean_permute_test_loss}')