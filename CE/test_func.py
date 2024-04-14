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
        # print("label.shape: ", label.shape)
        # print("\n\n", label)
        label = label.detach().cpu().numpy()
        label = label_to_vector(label) 
        print("label: ", label, "\n")

        pred = model(data)
        # print("pred.shape: ", pred.shape)
        # print("\n\n", pred)
        pred = pred.detach().cpu().numpy()
        pred = pred_to_vector(pred) 
        print("pred: ", pred, "\n")

        correct_pred += (label == pred).sum() # number of correct predictions
        # print("correct_pred: ", correct_pred)
        num_test += len(label) # add number of tests each iteration

    return correct_pred / num_test if num_test != 0 else 0 # prevent divide by 0 


def label_to_vector(label):
    '''
    Input: label --> numpy matrix of size B x sample_size
    Return: vector of size B with each element the average of input at the corresponding index 
    '''
    result = np.mean(label, axis=1) 
    return np.round(result)


def pred_to_vector(pred):
    '''
    Input: pred --> numpy matrix of size B x num_classes
    Return: vector of size B with each element the index of higher values iterate through pred 
    '''
    result = []
    for i in range(len(pred)):
        if pred[i][0] > pred[i][1]:
            result.append(0)
        else:
            result.append(1)

    return np.array(result)