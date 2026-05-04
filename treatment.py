import torch
from torch import nn, Tensor
import torch.nn.functional as F
from loader import treatment_one_hot, treatment, covars, row_count
from seed import seed_everything
import matplotlib.pyplot as plt

torch.set_printoptions(sci_mode=False)

hidden_layer_count = 4
hidden_layer_size = 100
class TreatmentModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = F.relu
        self.input_size = covars.shape[1]
        self.init_layer = nn.Linear(self.input_size, hidden_layer_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layer_count):
            self.hidden_layers.append(nn.Linear(hidden_layer_size,hidden_layer_size))
        self.final_layer = nn.Linear(hidden_layer_size, 4)
    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.init_layer(x))
        for i in range(hidden_layer_count):
            h = self.hidden_layers[i]
            x = self.activation(h(x))
        x = self.final_layer(x)
        return x
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)

step_count = 40
trial_count = 50
test_losses = torch.zeros(trial_count, step_count)
test_probs = torch.zeros(trial_count, row_count, step_count, 4)

fold_count = 5
print('Train Treatment Model')
for trial in range(trial_count):
    print(f'trial {trial + 1}',end='',flush=True)
    # seed_everything(trial)
    chunks = torch.chunk(torch.randperm(row_count), fold_count)
    for fold in range(fold_count):
        print('.',end='',flush=True)
        test_rows = chunks[fold].tolist()
        train_rows = torch.cat([chunks[i] for i in range(fold_count) if i != fold])
        test_target = treatment[test_rows]
        test_covars = covars[test_rows,:]
        train_target = treatment[train_rows]
        train_covars = covars[train_rows,:]
        model = TreatmentModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        for step in range(step_count):
            train_output = model(train_covars)
            train_loss = F.cross_entropy(train_output, train_target)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                test_output = model(test_covars)
                test_loss = F.cross_entropy(test_output, test_target)
                test_probs[trial,test_rows,step,:] = torch.softmax(test_output,dim=1)
                test_losses[trial,step] = test_loss.flatten()
                mean_test_loss = torch.mean(test_loss).item()
    print('')
print('Training Complete')

loss_path = torch.mean(test_losses,dim=0)
stop_time = int(torch.argmin(loss_path).item())
min_loss = torch.min(loss_path).item()
print(f'stop time {stop_time}')
print(f'min test loss {min_loss}')

outputs = torch.mean(test_probs[:,:,stop_time,:],dim=0)
treatment_file = open('output/treatment.csv', mode='w', buffering=1)
_ = treatment_file.write('treatment0,treatment1,treatment2,treatment3,prob0,prob1,prob2,prob3\n')
for row in range(row_count):
    s = ''
    s += f'{treatment_one_hot[row,0].int()},'
    s += f'{treatment_one_hot[row,1].int()},'
    s += f'{treatment_one_hot[row,2].int()},'
    s += f'{treatment_one_hot[row,3].int()},'
    s += f'{outputs[row,0]},{outputs[row,1]},{outputs[row,2]},{outputs[row,3]}'
    _ = treatment_file.write(f'{s}\n')

plt.clf()
plt.plot(loss_path.tolist())
plt.axvline(stop_time,color='blue')
plt.savefig('plots/treatment_loss.pdf')
plt.clf()

