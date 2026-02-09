from math import sqrt
from numpy import tri
import torch
from torch import LongTensor, nn, Tensor
import torch.nn.functional as F
from torch.utils.data import random_split, TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt

from loader import target, treatment, covars

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

# TO DO:
# measure treatment effects using counterfactal predictions
# permutation test to identify significance
# reconsider which variables to include (include only relevant variables)

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

def get_observation_path(observation: int, trial_count = 10, step_count= 50):
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
    observation_loss_path = torch.mean(test_loss_path,dim=1)
    observation_output_path = torch.mean(output_path,dim=1)
    return observation_output_path, observation_loss_path

trials = 2
step_count = 40
test_loss_matrix = torch.zeros(step_count, len(dataset)).to(device)
output_matrix = torch.zeros(step_count, len(dataset), 4).to(device)
for observation in range(len(dataset)):
    output_path, test_loss_path = get_observation_path(observation,trials,step_count)
    stop_time = torch.argmin(test_loss_path,dim=0)
    test_loss_matrix[:,observation] = test_loss_path
    output_matrix[:,observation,:] = output_path
    print(f'observation: {observation+1} / {len(dataset)}, stop_time: {stop_time}')
print(test_loss_matrix)
mean_loss_path = torch.mean(test_loss_matrix,dim=1).cpu().numpy()
mean_output_path = torch.mean(output_matrix,dim=1).cpu().numpy()
mean_output_path[-1,:]

plt.plot(mean_loss_path)
plt.show()
