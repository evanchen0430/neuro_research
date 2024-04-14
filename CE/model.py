import torch.nn as nn

class CNN(nn.Module):
    # out_features = 16 for comp
    def __init__(self, num_channels, num_classes):
        super().__init__()

        # input shape: [batch_size, 16, 100]

        # Global branch
        self.upsample = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=2, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            # nn.Dropout1d(),
            nn.BatchNorm1d(128), 
            nn.Upsample(256), 
        )

        self.conv_1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout1d(),
            nn.BatchNorm1d(64), 
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout1d(),
            nn.BatchNorm1d(32), 
        )

        self.fc = nn.Sequential(
            nn.Flatten(), # [B, 320]
            nn.Linear(320, 80),
            nn.LeakyReLU(),
            nn.Linear(80, 32),
            nn.LeakyReLU(),
            nn.Linear(32, num_classes),
            nn.LeakyReLU(),
        )




    def forward(self, x):
        # print("\nInput Shape: ", x.shape, "\n")
        x = self.upsample(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.fc(x)
        # print("\nShape: ", x.shape, "\n")
        return x