import torch
import numpy as np


def test_model(model, data_loader, device):
    model.to(device)
    model.eval()
    correct_pred = 0
    num_test = 0
    for data, label in data_loader:
        data = data.to(device)
        label = label.to(device) # shape: [B x sample_size]
        label = label.detach().cpu().numpy()
        label = convert_vec(label) 
        print("label: ", label, "\n")

        pred = model(data)
        pred = pred.detach().cpu().numpy()
        pred = convert_vec(pred) 
        print("pred: ", pred, "\n")

        correct_pred += (label == pred).sum() # number of correct predictions
        # print("correct_pred: ", correct_pred)
        num_test += len(label) # add number of tests each iteration

    return correct_pred / num_test if num_test != 0 else 0 # prevent divide by 0 


def convert_vec(pred):
    '''
    Input: pred --> numpy matrix of size B x sample_size
    Return: vector of size B with each element the average of input at the corresponding index 
    '''
    result = np.mean(pred, axis=1) 
    return np.round(result)