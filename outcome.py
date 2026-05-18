import torch
from torch import nn, Tensor
import torch.nn.functional as F
from load import outcome, covars, obs_count
from params import hidden_layer_count, hidden_layer_size, step_count, trial_count
from seed import seed_everything
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = F.silu
        self.input_size = covars.shape[1]
        self.project_layer = nn.Linear(self.input_size, hidden_layer_size)
        self.extra_hidden_layers = nn.ModuleList()
        for _ in range(hidden_layer_count-1):
            self.extra_hidden_layers.append(nn.Linear(hidden_layer_size,hidden_layer_size))
        self.final_layer = nn.Linear(hidden_layer_size, 1)
    def forward(self, x: Tensor) -> Tensor:
        x = self.project_layer(x)
        for i in range(hidden_layer_count-1):
            h = self.extra_hidden_layers[i]
            x = x + self.activation(h(x))
        x = self.final_layer(x)
        return x
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)

test_losses = torch.zeros(trial_count, obs_count, step_count)
test_outputs = torch.zeros(trial_count, obs_count, step_count)

fold_count = obs_count
print('Train Outcome Model')
for trial in range(trial_count):
    print(f'trial {trial + 1}',end='',flush=True)
    # seed_everything(trial)
    chunks = torch.chunk(torch.randperm(obs_count), fold_count)
    for fold in range(fold_count):
        print('.',end='',flush=True)
        test_obs = chunks[fold].tolist()
        train_obs = torch.cat([chunks[i] for i in range(fold_count) if i != fold])
        test_target = outcome[test_obs,:]
        test_covars = covars[test_obs,:]
        train_target = outcome[train_obs,:]
        train_covars = covars[train_obs,:]
        model = Model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        for step in range(step_count):
            train_output = model(train_covars)
            train_loss = torch.mean((train_target - train_output) ** 2)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                test_output = model(test_covars)
                test_outputs[trial,test_obs,step] = test_output.flatten()
                test_losses[trial,test_obs,step] = ((test_target - test_output) ** 2).flatten()
    print('')
print('Training Complete')

loss_path = torch.mean(test_losses,dim=(0,1))
stop_time = int(torch.argmin(loss_path).item())
min_loss = torch.min(loss_path).item()
print(f'stop time {stop_time}')
print(f'min test loss {min_loss}')

outputs = torch.mean(test_outputs[:,:,stop_time],dim=0)
targets = outcome.flatten()
outcome_file = open('output/outcome.csv', mode='w', buffering=1)
_ = outcome_file.write('outcome,output\n')
for row in range(obs_count):
    _ = outcome_file.write(f'{targets[row]},{outputs[row]}\n')

plt.clf()
plt.plot(loss_path.tolist())
plt.axvline(stop_time,color='blue')
loss0 = loss_path.tolist()[0]
lossMin = min(loss_path.tolist())
lossRange = max(0.01,loss0-lossMin)
plt.ylim(top=loss0 + 0.1*lossRange,bottom=lossMin-0.1*lossRange)
plt.savefig('plots/outcome_loss.pdf')
plt.clf()
