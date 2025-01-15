import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler

df_perm = np.load('data/data_500_128x128.npy')
df_perm_f = np.reshape(df_perm, (df_perm.shape[0], -1))

class AutoEncoder(nn.Module):
    def __init__(self, layers=[4, 16, 64]):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            self.conv_block(1, layers[0]),
            self.conv_block(layers[0], layers[1]),
            self.conv_block(layers[1], layers[2]),
        )
        self.decoder = nn.Sequential(
            self.deconv_block(layers[2], layers[1]),
            self.deconv_block(layers[1], layers[0]),
            self.deconv_block(layers[0], 1),
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

scaler = MinMaxScaler()
scaler.fit(df_perm_f)
df_perm_norm = scaler.transform(df_perm_f).reshape(df_perm.shape)
X_dataset = torch.tensor(np.expand_dims(df_perm_norm,1), dtype=torch.float32)
X_train, X_valid = random_split(X_dataset, [450, 50])
trainloader = DataLoader(X_train, batch_size=16, shuffle=True)
validloader = DataLoader(X_valid, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder(layers=[16, 64, 256]).to(device)
optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
criterion = nn.MSELoss().to(device)

epochs = 200
train_loss, valid_loss = [], []
for epoch in range(epochs):
    # training
    epoch_train_loss = []
    for i, x in enumerate(trainloader):
        x = x.to(device)
        optimizer.zero_grad()
        y = model(x)
        loss = criterion(y, x)
        loss.backward()
        optimizer.step()
        epoch_train_loss.append(loss.item())
    train_loss.append(np.mean(epoch_train_loss))
    # validation
    model.eval()
    epoch_valid_loss = []
    with torch.no_grad():
        for i, x in enumerate(validloader):
            x = x.to(device)
            y = model(x)
            loss = criterion(y, x)
            epoch_valid_loss.append(loss.item())
    valid_loss.append(np.mean(epoch_valid_loss))

pred = model(X_dataset.to(device)).cpu().detach().numpy().squeeze()
xhat = scaler.inverse_transform(pred.reshape(df_perm_f.shape)).reshape(df_perm.shape)