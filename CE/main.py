import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from data_loader import dataset
from model import CNN
from train import train_model
from test_func import test_model
import numpy as np
import os

def train(model, batch_size, device, epochs, train_weight_name, log_file_name):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-2)
    train_data = dataset() 

    train_data_load = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    train_model(model, train_data_load, loss, optimizer, epochs, device, \
                train_weight_name, log_file_name)


def test(model, device, batch_size, test_score_log):
    SCORE_FILE = open(test_score_log, "w")
    test_data = dataset(train=False)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

    accuracy = test_model(model, test_data_loader, device)

    print("Accuracy: {:.6f}" .format(accuracy))
    SCORE_FILE.write("Accuracy: {:.6f}\n" .format(accuracy))
    SCORE_FILE.close()

if __name__ == "__main__":
    batch_size = 8 
    num_epochs = 20
    exp_name = "{}_epo_trial_1" .format(num_epochs)
    # exp_name = "test" 

    model = CNN(num_channels=16, num_classes=2)

    cuda = "cuda:0"
    device=torch.device(cuda) 

    weight_dir = "weight"
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    train_weight_name = weight_dir + "/" + exp_name + ".pt"
    log_file_name = log_dir + "/" + exp_name + ".txt"

    # train model
    train(model, batch_size, device, num_epochs, train_weight_name, log_file_name)


    # test accuracy 
    # model.load_state_dict(torch.load(train_weight_name))
    # model.to(device)

    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    test_score_log = result_dir + "/" + exp_name + ".txt" 
    test(model, device, batch_size, test_score_log)


   
