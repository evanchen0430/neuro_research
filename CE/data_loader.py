import numpy as np
import os
import datetime
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

data_path = "data/" # replace with data path  

class dataset(Dataset):
    def __init__(self, data_dir=data_path, time_stamp_offset=3, train=True, num_channels=16, train_split=0.8, \
                 permute_seed=42, sample_size=100):
        self.data_dir = data_dir

        # data shape: (total data / sample size) x 16 x sample_size
        # label shape: (total data / sample size) x sample_size
        self.data, self.label = self.gen_data(data_dir, num_channels, time_stamp_offset=time_stamp_offset, sample_size=sample_size)
        # print("self data shape: ", self.data.shape)
        # print("self label shape: ", self.label.shape)
        self.train = train
        self.train_split = train_split

        self.num_data = len(self.label)
        # print("self num_data: ", self.num_data)
        self.split = int(train_split * self.num_data)

        # Randomly shuffle the indices
        self.indices = np.random.RandomState(permute_seed).permutation(self.num_data)

        if self.train:
            self.indices = self.indices[:self.split]
        else:
            self.indices = self.indices[self.split:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        return torch.from_numpy(self.data[idx]).to(torch.float), torch.from_numpy(self.label[idx]).to(torch.float)

    def gen_data(self, directory_path, num_channels, time_stamp_offset, sample_size):
        '''
        Loads data from the files and returns training and testing data
        @param filepath: The path to the data files
        @param num_channels: The number of channels in the data
        @param data_points_per_trial: The number of data points per trial
        @param training_split: The percentage of data to be used for training
        @param time_stamp_offset: stress state start time (minutes)
        @param sample_size: interval to sample EEG data
        @return: 

        Note:
        The data needs to be in the following format: first 16 lines are the data from the 16 channels, the last line is the time stamp
        Directory path is the path of where all the data is located
        '''

        all_data = []
        all_time_stamps = []
        
        # get all data and time stamps 
        for filename in os.listdir(directory_path):
            with open(os.path.join(directory_path, filename), 'r') as EEG_file:
                EEG_data = [[] for _ in range(16)]
                time_stamps = []
                start_reading = False
                for line in EEG_file:
                    if line.strip():
                        parts = line.split()
                    if parts[0] == "Sample":
                        start_reading = True
                    elif start_reading:
                        # each index represents each EEG signal (idx 0: fist EEG)
                        for i in range(num_channels): 
                            EEG_data[i].append(float(parts[i + 1][:-2]))
                        time_stamps.append(parts[-1])
                all_data.append(EEG_data)
                all_time_stamps.append(time_stamps)

        # trim off begin and end of each data to match the minimum length
        min_length = min([len(i) for i in all_time_stamps])
        for data_idx, data in enumerate(all_data):
            offset = len(data[0]) - min_length
            max_length = len(data[0]) - offset//2
            for i in range(num_channels):
                data[i] = data[i][offset//2:max_length]
            all_time_stamps[data_idx] = all_time_stamps[data_idx][offset//2:max_length]
            all_data[data_idx] = data


        # Generate labels based on time stamp and data collection offset (0: not stressed, 1: stressed)
        all_labels = []
        for time in all_time_stamps: 
            start_time = datetime.datetime.strptime(time[0],"%H:%M:%S.%f")
            offset_time = datetime.timedelta(hours=0, minutes=time_stamp_offset) 
            offset_time = start_time + offset_time
            labels = []
            for t in time: 
                if datetime.datetime.strptime(t, "%H:%M:%S.%f") > offset_time:
                    labels.append(1)
                else:
                    labels.append(0)
            
            all_labels.append(labels)


        # Sample data and labels based on sample_size
        new_data = []
        new_label = []
        for num_file, data in enumerate(all_data):
            sampled_data = []
            sampled_labels = []
            for i in range(len(all_labels[0])//sample_size):
                sampled_channel_data = []
                for channel_data in data:
                    sampled_channel_data.append(channel_data[i*sample_size:(i+1)*sample_size])
                sampled_data.append(sampled_channel_data)
                sampled_labels.append(all_labels[num_file][i*sample_size:(i+1)*sample_size])

            # concatenate lists
            new_data += sampled_data
            new_label += sampled_labels
        
        data = np.array(new_data)
        label = np.array(new_label)
        # print(data.shape)
        # print(label.shape)

        return data, label


if __name__ == "__main__":
    path = "data/"
    data_loader = dataset()
    # data, label = data_loader[0]
    for data, label in data_loader: 
        print(label)
        print(label.shape, "\n\n")

    # print(data.shape)
    # print(label.shape)
