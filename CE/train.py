import torch
from tqdm import tqdm
import numpy as np

def train_model(model, data_loader, loss, optimizer, epochs, device, train_weight_name, log_file_name):
	LOG_FILE = open(log_file_name, "w")
	model.to(device)
	model.train()
	for epoch in range(1, epochs+1):
		print("Epoch: ", epoch)
		loss_list = []
		for data, label in tqdm(data_loader):
			data = data.to(device)
			label = label.to(device) 
			label = convert_CE_label(label, device)
			pred = model(data)
			pred = pred.to(device)

			optimizer.zero_grad()
			# print("pred shape: ", pred.shape)
			# print("label shape: ", label.shape)
			# print(label)
			batch_loss = loss(pred, label)
			batch_loss.backward()
			optimizer.step()
			loss_list.append(batch_loss.cpu().detach().numpy())
		print("Loss: ", sum(loss_list)/len(loss_list)) # Prints value of loss
		
		torch.save(model.state_dict(), train_weight_name)
		LOG_FILE.write("Epoch {:2d}, Loss: {:.6f}\n" .format(epoch, sum(loss_list)/len(loss_list)))
		LOG_FILE.flush()
	
	LOG_FILE.close()


def convert_CE_label(label, device):
	'''
	Input: label --> tensor
	Return: 1d np array with each element the sum of each row
	'''
	label = label.detach().cpu().numpy() # convert tensor to numpy
	result = []
	for i in range(len(label)):
		result.append(label[i][0]) # same value in each vector in label

	result = torch.tensor(np.array(result))
	result = result.type(torch.LongTensor)
	return result.to(device)