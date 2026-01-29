from math import sqrt
import torch
from torch import LongTensor, nn, Tensor
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
# produce out of sample predictions

# TO DO:
# identify the early stopping time (i.e. batch count)
# measure treatment effects using counterfactal predictions
# reconsider which variables to include (include only relevant variables)

input_size = treatment.shape[1] + covars.shape[1]
hidden_count = 4
hidden_width = 100

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

dataset = TensorDataset(target, treatment, covars)

def get_counterfactual_predictions(observation: int, trials = 100, batch_count= 50):
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
    outputs = torch.zeros(4, trials).to(device)
    for trial in range(trials):
        model = Model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        for batch in range(batch_count):
            (train_target, train_treatment, train_covars) = next(iter(train_dataloader))
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
            # print(f'batch: {batch}, loss: {train_loss_value:.5f}, test: {test_loss_value:.5f}')
        (test_target, test_treatment, test_covars) = next(iter(test_dataloader))
        test_target: Tensor = test_target
        test_covars: Tensor = test_covars
        x = Tensor(range(4)).long()
        treatments = F.one_hot(Tensor(range(4)).long()).to(device)
        targets = test_target.repeat(4,1).to(device)
        covars = test_covars.repeat(4,1).to(device)
        output = model(treatments, covars)
        outputs[:,trial] = output[:,0]
    predictions = torch.mean(outputs,dim=1)
    return predictions

trials = 20
batch_count = 20
predictions = torch.zeros(len(dataset),1)
treatment_numbers = torch.argmax(treatment,1)
for i in range(len(dataset)):
    print(f'observation {i+1} / {len(dataset)}')
    counterfactual_predictons = get_counterfactual_predictions(i,trials,batch_count)
    predictions[i,0] = counterfactual_predictons[treatment_numbers[i]]
loss = F.mse_loss(predictions, target)