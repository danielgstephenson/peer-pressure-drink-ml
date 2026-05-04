import torch
from torch import nn, Tensor
import torch.nn.functional as F
from loader import covars, row_count
from residual import outcome_residual, treatment_residual
from params import hidden_layer_count, hidden_layer_size, step_count, trial_count
from seed import seed_everything
import matplotlib.pyplot as plt

torch.set_printoptions(sci_mode=False)

class CateModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = F.relu
        self.input_size = covars.shape[1]
        self.init_layer = nn.Linear(self.input_size, hidden_layer_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layer_count):
            self.hidden_layers.append(nn.Linear(hidden_layer_size,hidden_layer_size))
        self.final_layer = nn.Linear(hidden_layer_size, 3)
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
test_outputs = torch.zeros(trial_count, row_count, step_count, 3)

fold_count = 5
print('Train CATE Model')
for trial in range(trial_count):
    print(f'trial {trial + 1}',end='',flush=True)
    # seed_everything(trial)
    chunks = torch.chunk(torch.randperm(row_count), fold_count)
    for fold in range(fold_count):
        print('.',end='',flush=True)
        test_rows = chunks[fold].tolist()
        train_rows = torch.cat([chunks[i] for i in range(fold_count) if i != fold])
        test_target = outcome_residual[test_rows,:]
        test_covars = covars[test_rows,:]
        test_treat_res = treatment_residual[test_rows,1:]
        train_target = outcome_residual[train_rows,:]
        train_covars = covars[train_rows,:]
        train_treat_res = treatment_residual[train_rows,1:]
        model = CateModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        for step in range(step_count):
            train_output = model(train_covars)
            train_prediction = torch.einsum('ij,ij->i',train_treat_res,train_output).unsqueeze(1)
            train_loss = F.mse_loss(train_prediction, train_target)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                test_output = model(test_covars)
                test_prediction = torch.einsum('ij,ij->i',test_treat_res,test_output).unsqueeze(1)
                test_loss = F.mse_loss(test_prediction, test_target)
                test_outputs[trial,test_rows,step,:] = test_output
                test_losses[trial,step] = test_loss.flatten()
    print('')
print('Training Complete')

loss_path = torch.mean(test_losses,dim=0)
stop_time = int(torch.argmin(loss_path).item())
min_loss = torch.min(loss_path).item()
print(f'stop time {stop_time}')
print(f'min test loss {min_loss}')

outputs = torch.mean(test_outputs[:,:,stop_time,:],dim=0)
outcome_file = open('output/cate.csv', mode='w', buffering=1)
_ = outcome_file.write('cate1,cate2,cate3\n')
for row in range(row_count):
    _ = outcome_file.write(f'{outputs[row,0]},{outputs[row,1]},{outputs[row,2]}\n')

plt.clf()
plt.plot(loss_path.tolist())
plt.axvline(stop_time,color='blue')
plt.ylim(top=loss_path.tolist()[0],bottom=min(loss_path.tolist()))
plt.savefig('plots/cate_loss.pdf')
plt.clf()
