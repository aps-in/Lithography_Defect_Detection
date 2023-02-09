
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
    
    # Load image paths for training
    train_img_paths = pd.read_csv('labels.csv').values
    np.random.shuffle(train_img_paths)
    
    # Load image paths for testing
    test_img_paths = pd.read_csv('test_labels.csv').values
    np.random.shuffle(test_img_paths)
    
    # Define the ratio of the dataset to be used for training
    train_test_split_ratio = 0.8
    
    # Determine the number of training data points
    num_train_points = int(len(data)*train_test_split_ratio)
    
    # Split the image paths into training and testing sets
    test_img_paths = data[num_train_points:]
    train_img_paths = data[:num_train_points]
    
    # Initialize the Model with CNN and LSTM layers
    model = Model(CNN(),LSTM(in_size=512))
    
    # Define the list of device IDs for parallel training
    device_ids = []
    
    # Define the training hyperparameters
    num_epochs = 50
    batch_size = 64
    learning_rate = 0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = scheduler.StepLR(optimizer = optimizer, step_size = 5, gamma = 0.5)
    
    # Initialize the trainer
    trainer = Trainer(path, model, train_img_paths, test_img_paths, batch_size, num_epochs,
                      criterion, optimizer, lr_scheduler)

    # Train the model
    trainer.train_model(device_ids)

if __name__ == '__main__':
    main()
