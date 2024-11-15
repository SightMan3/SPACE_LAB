import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error


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


def TrainNN(model, x, y, loss_fn, n_epochs, alpha):
    curr_loss = None
    history = []
    optim = torch.optim.Adam(model.parameters(), lr=alpha)

    for i in range(n_epochs):
        model.train()
        for x_batch, y_batch in zip(x, y):
            for batch, y_real in zip(x_batch, y_batch):

                batch = batch.to(torch.float32)
                y_real = y_real.to(torch.float32)

                y_pred  = model(batch)
                loss    = loss_fn(y_pred, y_real)
                curr_loss = loss

                optim.zero_grad()
                loss.backward()

                optim.step()

        print(f"epoch n: {i+1}, Loss: {curr_loss}")
        history.append(curr_loss)

    return history


def PredictEvent(model, ev_num, events):
    model.eval()

    x, y = events

    preds = []
    for batch in x[ev_num]:
        batch = batch.float()
        preds.append(model(batch).detach().numpy())
    preds = np.array(preds)

    plt.plot(preds, label='prediction')
    plt.plot(y[ev_num], label='real')
    plt.legend()
    plt.show()


def PredictPeriod(model, period):
    model.eval()
    x, y = period
    preds = []

    for batch in x:
        batch = batch.float()
        preds.append(
            model(batch).detach().numpy()
        )

    preds = np.array(preds)

    plt.figure(figsize=(25, 5))
    plt.plot(preds, label='prediction')
    plt.plot(y, label='real')
    plt.legend()
    plt.show()

    return preds


if __name__ == '__main__':
    x_train, y_train = DataProcessing()

    model = nn.Sequential(
        nn.Linear(32, 26),
        nn.Tanh(),
        nn.Linear(26, 1),
    )

    # torch.save(model.state_dict(), "./data/xxx.pth")

    _ = TrainNN(model,
        x_train, y_train,
        nn.MSELoss(),
        n_epochs=40,
        alpha=0.0003,
    )

    torch.save(model.state_dict(), './data/FFNN.pth')

    PredictEvent(model, ev_num=45, events=(x_train, y_train))

    valid_data = pd.read_csv('./data/test_dst_new.csv')
    valid_data.set_index('index', inplace=True)
    valid_data.index = pd.to_datetime(valid_data.index)

    x_valid_79, y_valid_79, x_valid_80, y_valid_80, x_valid_81, y_valid_81 = ValidationDataProcessing(valid_data)

    preds_79 = PredictPeriod(model, (x_valid_79, y_valid_79))
    preds_80 = PredictPeriod(model, (x_valid_80, y_valid_80))
    preds_81 = PredictPeriod(model, (x_valid_81, y_valid_81))

    
    Predictions = np.concatenate((preds_79, preds_80, preds_81))
    Predictions = pd.DataFrame(data={
        'DST_prediction_1h': Predictions.flatten()
    })

    stripped_valid = pd.concat([
        valid_data['Dst_index'][valid_data.index.year == 1979].iloc[8:],
        valid_data['Dst_index'][valid_data.index.year == 1980].iloc[8:],
        valid_data['Dst_index'][valid_data.index.year == 1981].iloc[8:]
    ])
    Predictions.index = stripped_valid.index

    Predictions.to_csv('./data/DST_prediction_Model_A.csv')



#%%
