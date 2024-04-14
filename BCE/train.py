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


def convert_pred(pred):
	'''
	Input: pred --> tensor
	Return: 1d np array with each element the sum of each row
	'''
	pred = pred.detach().cpu().numpy() # convert tensor to numpy
	result = []
	for row in range(len(pred)):
		sum_row = 0
		# max_val = pred[row][0] # initialize first element with highest probability
		# max_idx = 0
		for i, val in enumerate(pred[row]):
			# if val > max_val: 
			# 	max_idx = i
			# 	max_val = val
			sum_row += val
		result.append(sum_row)
		# result.append(max_idx)
		# print(max_val, "\n")
		# result.append(max_val)
	
	return np.array(result)