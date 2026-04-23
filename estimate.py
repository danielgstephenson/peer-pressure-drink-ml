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

step_count = 40
trial_count = 100

dataset = TensorDataset(target, treatment, covars)
potential_treatments = torch.eye(4).to(device)

estimate_file = open('output/estimates.csv', mode='a', buffering=1)
_ = estimate_file.write('trial,observation,treatment1,treatment2,treatment3,treatment4\n')

for trial in range(trial_count):
    print(f'trial {trial + 1}')
    _ = torch.manual_seed(trial)
    torch.use_deterministic_algorithms(True)
    for observation in range(len(dataset)):
        if observation % 2 == 0: print('.',end='',flush=True)
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
            test_loss = F.mse_loss(test_output, test_target).item()
            repeat_covars = test_covars.repeat([4,1])
            outputs = model(potential_treatments, repeat_covars).squeeze(1).tolist()
            _ = estimate_file.write(f'{trial},{observation},{outputs[0]},{outputs[1]},{outputs[2]},{outputs[3]}\n')
    print('')
            

