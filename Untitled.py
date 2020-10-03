#!/usr/bin/env python
# coding: utf-8




import copy
import csv
import os
import datetime
import collections
import random
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from nltk.stem import SnowballStemmer
from scipy.sparse import dok_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay





path = '/home/samsdimko/lab1_texts/'
name_classes = [name[:-4] for name in os.listdir(path)]
print(name_classes)




texts = list()

for names in os.listdir(path):
    with open(path+names, newline='') as csvfile:
        reader = csv.reader(csvfile)
        heads = next(reader)
        text = np.array(list(reader))[:,4]
        texts.append(text)
texts = np.array(texts)

train_data = list()
test_data = None
train_label = list()
test_label = list()

for i in range(np.shape(texts)[0]):
    test_len = len(texts[i]) // 5
    label = np.zeros(shape=test_len)    
    label.fill(i)
    test_label.append(label)
    if type(test_data) == type(None):
        test_data = np.array(np.random.choice(texts[i],test_len, replace=False))
    else:
        test_data = np.concatenate([test_data, np.array(np.random.choice(texts[i],test_len, replace=False))])
    for j in range(len(texts[i])):
        if not (texts[i][j] in test_data):
            train_data.append(texts[i][j])
            train_label.append(i)

    
test_label = np.array(test_label)
test_label = np.concatenate([np.array(x) for x in test_label])
train_data = np.array(train_data)
train_label = np.array(train_label)


token_search = re.compile(r'[\w\d]+')
stemmer = SnowballStemmer("russian")

train_data = [token_search.findall(data) for data in train_data]
train_data = np.array([np.array(x) for x in train_data])
test_data = [token_search.findall(data) for data in test_data]
test_data = np.array([np.array(x) for x in test_data])

def stems(data):
    for i in range(len(data)):
         for j in range(len(data[i])):
            data[i][j] = stemmer.stem(data[i][j]) 
    return data

train_data = stems(train_data)
test_data = stems(test_data)



def get_vocab(data):
    vocab = set()
    for x in data:
        for y in x:
            vocab.add(y)
    vocab = list(vocab)
    vocab_freq = collections.defaultdict(float)

    for text in data:
        txt = set(text)
        for word in txt:
            vocab_freq[word] += 1
    vocab_freq = {word: freq for (word, freq) in vocab_freq.items()
                  if freq / len(data) <= 0.8 and freq > 4}
    vocab_freq = dict(sorted(vocab_freq.items(),
                       reverse=True,
                       key=lambda pair: pair[1]))
    for word in vocab_freq:
        vocab_freq[word] = np.log(len(data) / vocab_freq[word])
    vocab = list(vocab_freq.keys())
    word_freq = np.zeros(shape=len(vocab))
    for i in range(len(vocab)):
        word_freq[i] = vocab_freq[vocab[i]]
    vocabulary = dict()
    for i in range(len(vocab)):
        vocabulary.update({vocab[i]: i})
    return vocab, vocab_freq, word_freq, vocabulary


vocab, vocab_freq, word_freq, vocabulary = get_vocab(train_data)




def TfIdf(data, vocabulary, vocab, word_freq):
    tf_idf = np.zeros((np.shape(data)[0], len(vocab)), dtype='float32')
    for i in range(len(data)):
        for word in data[i]:
            if word in vocabulary:
                tf_idf[i][vocabulary[word]] += 1
    tf_idf = dok_matrix(tf_idf)
    tf_idf = tf_idf.tocsr()
    tf_idf = tf_idf.multiply(1 / tf_idf.sum(1))
    tf_idf = tf_idf.multiply(word_freq)
    tf_idf = tf_idf.tocsc()
    tf_idf /= tf_idf.max()
    return tf_idf.tocsr()

vector_train = TfIdf(train_data, vocabulary, vocab, word_freq)
vector_test = TfIdf(test_data, vocabulary, vocab, word_freq)




class MyDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        cur_features = torch.from_numpy(self.features[idx].toarray()[0]).float()
        cur_label = torch.from_numpy(np.asarray(self.targets[idx])).long()
        return cur_features, cur_label
    
train_dataset = MyDataset(vector_train, train_label)
test_dataset = MyDataset(vector_test, test_label)



model = nn.Linear(len(vocabulary), np.shape(texts)[0])
scheduler = lambda optim:     torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5, verbose=True)

epoch_n = 200
lr = 1e-1

shuffle_train=True

data_loader_ctor=DataLoader

device = 'cuda'

max_batches_per_epoch_train=10000
max_batches_per_epoch_val=1000


device = torch.device(device)
model.to(device)

early_stopping_patience=10
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)


lr_scheduler = scheduler(optimizer)

batch_size = 32
criterion=F.cross_entropy
dataloader_workers_n = 0
train_dataloader = data_loader_ctor(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                    num_workers=dataloader_workers_n)
val_dataloader = data_loader_ctor(test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=dataloader_workers_n)

best_val_loss = float('inf')
best_epoch_i = 0
best_model = copy.deepcopy(model)

for epoch_i in range(epoch_n):
    epoch_start = datetime.datetime.now()
    print('Эпоха {}'.format(epoch_i))

    model.train()
    mean_train_loss = 0
    train_batches_n = 0
    for batch_i, (batch_x, batch_y) in enumerate(train_dataloader):
        if batch_i > max_batches_per_epoch_train:
            break

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        pred = model(batch_x)
        loss = criterion(pred, batch_y)

        model.zero_grad()
        loss.backward()

        optimizer.step()

        mean_train_loss += float(loss)
        train_batches_n += 1

    mean_train_loss /= train_batches_n
    print('Эпоха: {} итераций, {:0.2f} сек'.format(train_batches_n,
                                                   (datetime.datetime.now() - epoch_start).total_seconds()))
    print('Среднее значение функции потерь на обучении', mean_train_loss)
    best_model = copy.deepcopy(model)


    model.eval()
    mean_val_loss = 0
    val_batches_n = 0

    with torch.no_grad():
        for batch_i, (batch_x, batch_y) in enumerate(val_dataloader):
            if batch_i > max_batches_per_epoch_val:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            mean_val_loss += float(loss)
            val_batches_n += 1

    mean_val_loss /= val_batches_n
    print('Среднее значение функции потерь на валидации', mean_val_loss)

    if mean_val_loss < best_val_loss:
        best_epoch_i = epoch_i
        best_val_loss = mean_val_loss
        best_model = copy.deepcopy(model)
        print('Новая лучшая модель!')
    elif epoch_i - best_epoch_i > early_stopping_patience:
        print('Модель не улучшилась за последние {} эпох, прекращаем обучение'.format(
            early_stopping_patience))
        break

    if lr_scheduler is not None:
        lr_scheduler.step(mean_val_loss)

    print()

print(best_val_loss, best_model)



def predict_with_model(model, dataset, device=None, batch_size=32, num_workers=0, return_labels=False):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_by_batch = []

    device = torch.device(device)
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels = []
    with torch.no_grad():
        import tqdm
        for batch_x, batch_y in tqdm.tqdm(dataloader, total=len(dataset)/batch_size):
            batch_x = batch_x.to(device)

            if return_labels:
                labels.append(batch_y.numpy())

            batch_pred = model(batch_x)
            results_by_batch.append(batch_pred.detach().cpu().numpy())

    if return_labels:
        return np.concatenate(results_by_batch, 0), np.concatenate(labels, 0)
    else:
        return np.concatenate(results_by_batch, 0)





train_pred = predict_with_model(best_model, train_dataset)

train_loss = F.cross_entropy(torch.from_numpy(train_pred),
                             torch.from_numpy(train_label).long())

print('Среднее значение функции потерь на обучении', float(train_loss))
print('Доля верных ответов', accuracy_score(train_label, train_pred.argmax(-1)))
print()



test_pred = predict_with_model(best_model, test_dataset)

test_loss = F.cross_entropy(torch.from_numpy(test_pred),
                            torch.from_numpy(test_label).long())

print('Среднее значение функции потерь на валидации', float(test_loss))
print('Доля верных ответов', accuracy_score(test_label, test_pred.argmax(-1)))




conf = confusion_matrix(y_true=test_label, y_pred=test_pred.argmax(-1))                      
disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=name_classes).plot(cmap ='inferno', xticks_rotation='vertical')
plt.show()

