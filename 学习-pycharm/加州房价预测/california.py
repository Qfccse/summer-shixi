import torch
import numpy as np
import pandas as pd
from torch import nn


test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")
print(test_data.shape)
print(train_data.shape)
print(train_data.dtypes)
print(test_data.dtypes)


all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))
idx = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[idx] = all_features[idx].apply(lambda x:(x-x.mean()/x.std()))
all_features[idx] = all_features[idx].fillna(0)

all_features = pd.get_dummies(all_features,dummy_na=True)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:,:n_train])
test_features = torch.tensor(all_features[:,n_train:])

loss = nn.MSELoss()

def get_net():
    net =  nn.Sequential(nn.Flatten(),
                         nn.Linear(n_train,256),
                         nn.ReLU(),
                         nn.Linear(256,1))
    for p in net.parameters():
        nn.init.normal_(p,mean=0,std=0.01)

    return net

def log_rmse(net,features,labels):
    with torch.no_grad():
        clipped_pred = torch.max(net(features),torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_pred.log(),labels.log()))

    return rmse.item()