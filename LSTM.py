import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def generate_batches(a, b, c, d, y, n):
    '''
    Function creates batches out of a, b, c, d data of size 8 hours,
    8-hour window is moved throughout the arrays with step one hour,
    Than all the 8-hour batches are concatenated together to create one big 32 size batch.
    Finally appended to main array.

    Also cuts first 8 hours out of y data, so that for every 8 hours of x data
    there is one hour of y data hour ahead of input neurons data.

    Parameters
    ----------
    a - Bz
    b - Sigma Bz
    c - n
    d = v
    y - y training data (DST)
    n - length of event

    Returns
    -------
    batches - Matrix where rows correspond to 8 hours of x array data
          y - y training data (DST) that has first 8 hours cut off
    '''

    y = y[8:]
    y = torch.from_numpy(np.array(y))

    batches = []

    for i in range(n):
        if (i+8) <= n:
            # print(i, i+8)
            batch_a = a[i:i+8]
            batch_b = b[i:i+8]
            batch_c = c[i:i+8]
            batch_d = d[i:i+8]

            final_batch = np.concatenate((batch_a, batch_b, batch_c, batch_d), axis=None)

            batches.append(final_batch)

    batches = batches[:-1]
    batches = torch.from_numpy(np.array(batches))

    return batches, y

def DataProcessing():
    '''
    Function Loops through trainind data in steps of 146. Because every event is 146 hours long.
    For every 146 hours of every parameter (Bz, sigma Bz, n, v, DST) is created a matrix with dimensions 146 X 32. of x data
    and vector of  size 146 values of y data. One y value for 32 values of parameters.
    This matrix is than added to final array. Which in the end corresponds to tensor of size 60 X 146 X 32 and matrix 60 X 146.
    60 Because there are 60 events in training data. And we need to have 32 values for every DST values so there is 146 X 32 matrix
    of parameters.

    Returns
    -------
    x_train data
    y_train data

    '''

    data = pd.read_csv('./data/train_dst_new.csv')

    x_train_batches = []
    y_train_batches = []

    for i in range(0, len(data), 147):
        Bz       = torch.from_numpy(data.loc[i:i+146]['Bz_GSE'][:-1].to_numpy())
        Bz_sigma = torch.from_numpy(data.loc[i:i+146]['Sigma_Bz_GSE'][:-1].to_numpy())
        n        = torch.from_numpy(data.loc[i:i+146]['Proton_density'][:-1].to_numpy())
        v        = torch.from_numpy(data.loc[i:i+146]['Plasma_speed'][:-1].to_numpy())
        DST      = torch.from_numpy(data.loc[i:i+146]['Dst_index'][:-1].to_numpy())

        x_train, y_train = generate_batches(Bz, Bz_sigma, n, v, DST, 146)

        y_train_batches.append(y_train)
        x_train_batches.append(x_train)

    return (x_train_batches, y_train_batches)


x_train, y_train = DataProcessing()
x_train = np.array(x_train)
y_train = np.array(y_train)

x_tra = x_train.reshape(60, 138, 4, 8)
x_tra = np.transpose(x_tra, (0, 1, 3, 2))
x_tra = torch.from_numpy(x_tra)

y_tra = y_train.reshape(60, -1, 1)
y_tra = torch.from_numpy(y_tra)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.rnn = nn.LSTM(
            input_size, hidden_size, num_stacked_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.rnn(x, (h0, c0))
        out    = self.fc(out[:, -1, :])
        return out
    


hidden_n  = 128
stacked_n = 3

model = LSTM(4, hidden_n, stacked_n)
model.to(device)


def train(epochs, x, y, loss_f, optimizer):
    for i in range(epochs):
        index = 0
        model.train(True)
        running_loss = 0.0

        print(f'EPOCH : {i}')
        
        for batch_x, batch_y in zip(x, y):
            index += 1
            
            batch_x = batch_x.to(torch.float32).to(device)
            batch_y = batch_y.to(torch.float32).to(device)

            output = model(batch_x)
            loss = loss_f(output, batch_y)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if index % 60 == 0:
                avg_loss_across_batches = running_loss / 100
                print(f'LOSS : {avg_loss_across_batches}')
                running_loss = 0.0

    print("DONE")

lr = 0.001
num_epochs = 75
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train(num_epochs, x_tra, y_tra, loss_function, optimizer)



def ValidationDataProcessing(valid_data):

    year_1979 = valid_data[valid_data.index.year == 1979]
    year_1980 = valid_data[valid_data.index.year == 1980]
    year_1981 = valid_data[valid_data.index.year == 1981]

    x_valid_79, y_valid_79 = generate_batches(
        year_1979['Bz_GSE'],
        year_1979['Sigma_Bz_GSE'],
        year_1979['Proton_density'],
        year_1979['Plasma_speed'],
        year_1979['Dst_index'],
        len(year_1979)
    )

    x_valid_80, y_valid_80 = generate_batches(
        year_1980['Bz_GSE'],
        year_1980['Sigma_Bz_GSE'],
        year_1980['Proton_density'],
        year_1980['Plasma_speed'],
        year_1980['Dst_index'],
        len(year_1980)
    )

    x_valid_81, y_valid_81 = generate_batches(
        year_1981['Bz_GSE'],
        year_1981['Sigma_Bz_GSE'],
        year_1981['Proton_density'],
        year_1981['Plasma_speed'],
        year_1981['Dst_index'],
        len(year_1981)
    )

    return x_valid_79, y_valid_79, x_valid_80, y_valid_80, x_valid_81, y_valid_81


valid_data = pd.read_csv('./data/test_dst_new.csv')
valid_data.set_index('index', inplace=True)
valid_data.index = pd.to_datetime(valid_data.index)

x_valid_79, y_valid_79, x_valid_80, y_valid_80, x_valid_81, y_valid_81 = ValidationDataProcessing(valid_data)

x_79 = x_valid_79.reshape(8349, 4, 8)
x_79 = torch.transpose(x_79, 2, 1)
x_79 = x_79.to(torch.float32).to(device)


model.eval()
preds = model(x_79)
preds_79 = preds.cpu().detach()

plt.figure(figsize=(15,6))
plt.plot(y_valid_79, label='real')
plt.plot(preds_79, label='prediction', c='orange')
plt.legend()