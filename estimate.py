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

def get_predictions(dataset: TensorDataset, step_count = 40, trial_count = 20) -> np.ndarray:
    test_loss_tensor = torch.zeros(step_count, trial_count, len(dataset)).to(device)
    output_tensor = torch.zeros(step_count, trial_count, len(dataset), 4).to(device)
    for observation in range(len(dataset)):
        output_path, test_loss_path = get_output_path(dataset, observation,trial_count,step_count)
        test_loss_tensor[:,:,observation] = test_loss_path
        output_tensor[:,:,observation,:] = output_path
        print('.',end='',flush=True)
    predictions = output_tensor[-1,:,:,:].cpu().numpy()
    return predictions

# Proposed Test Statistic: Count the number of individuals with a positive causal impact estimate

step_count=40
permute_count=5
trial_count=5

# original_dataset = TensorDataset(target, treatment, covars)
# original_predictions = get_predictions(original_dataset, step_count, trial_count)
# original_estimate_file = open('output/original_estimates.csv', mode='a', buffering=1) 
# original_sign_count_file = open('output/original_sign_count.csv', mode='a', buffering=1) 
# original_predictions = np.mean(original_predictions,0)
# original_effect_estimates = original_predictions[:,3] - original_predictions[:,2]
# original_effect_string = ",".join([str(x) for x in original_effect_estimates])
# original_estimate_file.write(original_effect_string+'\n')
# original_sign_count = np.sum(np.sign(original_effect_estimates))
# original_sign_count_file.write(f'{original_sign_count}\n')

# DEVELOP: def get_shuffled_effect_estimates(step_count=40, permute_count=5000) -> np.ndarray:
permute_sign_counts = np.zeros(permute_count)
permute_estimate_file = open('output/permute_estimates.csv', mode='a', buffering=1) 
permute_sign_count_file = open('output/permute_sign_count.csv', mode='a', buffering=1) 
print(f'Begin Permutation Test')
for i in range(permute_count):
    permute_dataset = get_permute_dataset()
    permute_predictions = get_predictions(permute_dataset, step_count, trial_count)
    permute_predictions = np.mean(permute_predictions,0)
    permute_effect_estimates = permute_predictions[:,3] - permute_predictions[:,2]
    permute_effect_string = ",".join([str(x) for x in permute_effect_estimates])
    permute_estimate_file.write(permute_effect_string+'\n')
    permute_sign_count = np.sum(np.sign(permute_effect_estimates))
    permute_sign_count_file.write(f'{permute_sign_count}\n')
    permute_sign_counts[i] = permute_sign_count
    print(f'Permutation Trial {i+1}: {permute_sign_count}')

# original_predictions = np.mean(original_predictions,0)
# original_effect_estimates = original_predictions[:,3] - original_predictions[:,2]
# plt.hist(original_effect_estimates)
# plt.show()

# >>> print(effect_estimates)
# [-0.11715002  0.09962441 -0.02398598  0.03438779 -0.1780526 ]
# [ 0.04274881 -0.04835736  0.05387061  0.1910139  -0.04945911]

# predictions = get_predictions(dataset, step_count=40, trial_count=5)
# trial_predictions = np.mean(predictions,1) 
# mean_predictions = np.mean(trial_predictions,0)
# std_predictions = np.std(trial_predictions,0)
# effect_estimate = mean_predictions[3] - mean_predictions[2]

# x_pos = np.arange(len(treatment_names))
# plt.bar(x_pos, mean_predictions, align='center', alpha=0.7)
# plt.axhline(0, color='black', linewidth=0.8)
# plt.xticks(x_pos, treatment_names)
# plt.ylabel('Treatment')
# plt.scatter(x_pos,mean_predictions+std_predictions)
# plt.scatter(x_pos,mean_predictions-std_predictions)
# plt.show()


