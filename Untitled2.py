#!/usr/bin/env python
# coding: utf-8

# In[171]:


import pandas as pd
import numpy as np
import csv
import math
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
import scikitplot
import matplotlib.pyplot as plt 

path = '/home/samsdimko/Documents/num_3/Tariff_plans_change.csv'
with open(path, newline='') as csvfile:
        reader = pd.DataFrame(csv.reader(csvfile))
        reader = reader[[0,1]][1:]
        reader = reader.to_numpy(dtype=float).T
train = []
label = []

for ids in range(np.shape(reader)[1]-1):
    if reader[0][ids] == reader[0][ids+1]:
        train.append(reader[1][ids])
        label.append(reader[1][ids+1])
print(len(train))
label = np.array(label)
label -= 1
train = np.array(train)
num_tariff = max(label)+1
print(num_tariff)
x = np.zeros(shape=int(num_tariff))
print(x)
x_train = []
y_train = []
for i in range(len(train)):
    x_train.append(np.zeros(shape=int(num_tariff)))
    y_train.append(np.zeros(shape=int(num_tariff)))
    x_train[i][int(train[i])-1] = 1                   
    y_train[i][int(label[i])-1] = 1
print(num_tariff)
x_train = np.array(x_train)
y_train = np.array(y_train)


# In[179]:


y_tensor = torch.from_numpy(y_train).float()
X_tensor = torch.from_numpy(x_train).float()
yy = torch.from_numpy(label).int()


# In[180]:


print(yy)
print(X_tensor)
print(type(X_tensor.shape[1]))


# In[181]:


epochs = 10000
learning_rate = 0.001
#Число итераций обучения и шаг обучения
class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, X):
        predictions = self.linear(X)
        return torch.sigmoid(predictions)
        #Метод обучения сигмоида

model = LogisticRegressionTorch(X_tensor.shape[1], X_tensor.shape[1])

batch_size = 6355
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#target = torch.empty(batch_size, dtype=torch.double).random_(nb_classes)
for epoch in range(epochs):
    labels = yy.long().squeeze_()
    optimizer.zero_grad()
    predictions = model(X_tensor)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()

predictions = model(X_tensor)


# In[182]:


print(predictions)


# In[ ]:




