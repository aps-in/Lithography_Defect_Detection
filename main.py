import numpy as np
import pandas as pd

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.utils.data as data

from train import Trainer
from model import Model, CNN, LSTM


def main():
    

    path = ''
    train_img_paths = pd.read_csv('labels.csv').values
    np.random.shuffle(train_img_paths)
    test_img_paths = pd.read_csv('test_labels.csv').values
    np.random.shuffle(test_img_paths)

    train_test_split_ratio = 0.8
    num_train_points = int(len(data)*train_test_split_ratio)
    test_img_paths = data[num_train_points:]
    train_img_paths = data[:num_train_points]
    

    model = Model(CNN(),LSTM(in_size=512))
    
    device_ids = []

    num_epochs = 50
    batch_size = 64
    learning_rate = 0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    lr_scheduler = scheduler.StepLR(optimizer = optimizer, step_size = 5, gamma = 0.5)

    trainer = Trainer(path, model, train_img_paths, test_img_paths, batch_size, num_epochs,
                      criterion, optimizer, lr_scheduler)


    trainer.train_model(device_ids)


if __name__ == '__main__':
    main()
