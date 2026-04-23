import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset
from loader import target, treatment, covars

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_printoptions(sci_mode=False)

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

def get_test_loss_path(dataset: TensorDataset, observation: int, step_count: int):
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
    test_loss_path = torch.zeros(step_count).to(device)
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
            test_loss_path[step] = test_loss_value
    return test_loss_path

def get_test_loss(dataset: TensorDataset, step_count: int) -> tuple[float, int]:
    test_loss_tensor = torch.zeros(len(dataset),step_count).to(device)
    for individual in range(len(dataset)):
        test_loss_path = get_test_loss_path(dataset, individual,step_count)
        test_loss_tensor[individual,:] = test_loss_path
        if individual % 2 == 0: print('.',end='',flush=True)
    mean_test_loss = torch.mean(test_loss_tensor,0)
    test_loss = float(torch.min(mean_test_loss).item())
    stop_time = int(torch.argmin(mean_test_loss).item())
    print('')
    return test_loss, stop_time

step_count = 100
trial_count = 100

original_dataset = TensorDataset(target, treatment, covars)
null_dataset = TensorDataset(target, 0*treatment, covars)
original_test_loss_file = open('output/original_test_loss.csv', mode='a', buffering=1)
null_test_loss_file = open('output/null_test_loss.csv', mode='a', buffering=1)

for trial in range(trial_count):
    original_test_loss, original_stop_time = get_test_loss(original_dataset, step_count)
    print(f'original: test loss {original_test_loss:0.4f}, stop time {original_stop_time}')
    original_test_loss_file.write(f'{original_test_loss},{original_stop_time}\n')
    null_test_loss, null_stop_time = get_test_loss(null_dataset, step_count)
    print(f'null: test loss {null_test_loss:0.4f}, stop time {null_stop_time}')
    null_test_loss_file.write(f'{null_test_loss},{null_stop_time}\n')

# The model makes more accurate predictions in the null_dataset
