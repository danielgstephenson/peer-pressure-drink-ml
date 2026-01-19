import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import random_split, TensorDataset, DataLoader, Subset
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

# TO DO:
# produce out of sample predictions
# identify the early stopping time
# measure treatment effects using counterfactal predictions
# reconsider which variables to include (include only relevant variables)

input_size = treatment.shape[1] + covars.shape[1]
hidden_count = 4
hidden_width = 100

n = target.shape[0]

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = F.relu
        self.init_layer = nn.Linear(input_size, hidden_width)
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_count):
            self.hidden_layers.append(nn.Linear(hidden_width,hidden_width))
        self.final_layer = nn.Linear(hidden_width, 1)
    def forward(self, treatment: Tensor, covars: Tensor) -> Tensor:
        x: Tensor = torch.cat([treatment, covars], dim=1)
        x = self.activation(self.init_layer(x))
        for i in range(hidden_count):
            h = self.hidden_layers[i]
            x = self.activation(h(x))
        x = self.final_layer(x)
        return x
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)

model = Model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

dataset = TensorDataset(target, treatment, covars)
train_dataset, test_dataset = random_split(dataset,[0.5, 0.5])

dataloader = DataLoader(
    train_dataset,
    batch_size=n,
    shuffle=True
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=n,
    shuffle=True
)

batch_data = next(iter(dataloader))

batch_count = 200
print('Training...')
for batch in range(batch_count):
    (train_target, train_treatment, train_covars) = next(iter(dataloader))
    train_output = model(train_treatment,train_covars)
    train_loss = F.mse_loss(train_output, train_target)
    train_loss_value = train_loss.detach().cpu().numpy()
    with torch.no_grad():
        (test_target, test_treatment, test_covars) = next(iter(test_dataloader))
        test_output = model(test_treatment, test_covars)
        test_loss = F.mse_loss(test_output, test_target)
        test_loss_value = test_loss.detach().cpu().numpy()
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f'batch: {batch}, loss: {train_loss_value:.5f}, test: {test_loss_value:.5f}')