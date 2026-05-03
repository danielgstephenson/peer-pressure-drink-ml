import torch
from torch import nn, Tensor
import torch.nn.functional as F
from loader import target, covars
from seed import seed_everything
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print("device = " + str(device))
torch.set_printoptions(sci_mode=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

hidden_layer_count = 4
hidden_layer_size = 100
row_count = covars.shape[0]
fold_count = 5
step_count = 200
trial_count = 5

class BaselineModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = F.relu
        self.input_size = covars.shape[1]
        self.init_layer = nn.Linear(self.input_size, hidden_layer_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layer_count):
            self.hidden_layers.append(nn.Linear(hidden_layer_size,hidden_layer_size))
        self.final_layer = nn.Linear(hidden_layer_size, 1)
    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.init_layer(x))
        for i in range(hidden_layer_count):
            h = self.hidden_layers[i]
            x = self.activation(h(x))
        x = self.final_layer(x)
        return x
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)

test_losses = torch.zeros(trial_count, step_count)
test_outputs = torch.zeros(trial_count, row_count, step_count)

print('Training...')
for trial in range(trial_count):
    print(f'trial {trial + 1}',end='',flush=True)
    seed_everything(trial)
    chunks = torch.chunk(torch.randperm(row_count), fold_count)
    for fold in range(fold_count):
        print('.',end='',flush=True)
        test_rows = chunks[fold].tolist()
        train_rows = torch.cat([chunks[i] for i in range(fold_count) if i != fold])
        test_target = target[test_rows,:]
        test_covars = covars[test_rows,:]
        train_target = target[train_rows,:]
        train_covars = covars[train_rows,:]
        model = BaselineModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        for step in range(step_count):
            train_output = model(train_covars)
            train_loss = F.mse_loss(train_output, train_target)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                test_output = model(test_covars)
                test_loss = F.mse_loss(test_output, test_target)
                test_outputs[trial,test_rows,step] = test_output.flatten()
                test_losses[trial,step] = test_loss.flatten()
                mean_test_loss = torch.mean(test_loss).item()
    print('')
print('Training Complete')

loss_path = torch.mean(test_losses,dim=0)
stop_time = int(torch.argmin(loss_path).item())

outputs = torch.mean(test_outputs[:,:,stop_time],dim=0)
targets = target.flatten()
baseline_file = open('output/baseline.csv', mode='w', buffering=1)
_ = baseline_file.write('observation,target,output\n')
for row in range(row_count):
    _ = baseline_file.write(f'{row+1},{targets[row]},{outputs[row]}\n')

plt.plot(loss_path.tolist())
plt.axvline(stop_time,color='blue')
plt.show()