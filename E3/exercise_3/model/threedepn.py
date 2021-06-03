import torch
import torch.nn as nn


class ThreeDEPN(nn.Module):
    def __init__(self):
        super(ThreeDEPN, self).__init__()

        self.num_features = 80

        # TODO: 4 Encoder layers
        self.conv1 = nn.Conv3d(2,self.num_features,4,stride=2,padding=1)
        self.conv2 = nn.Conv3d(self.num_features,self.num_features*2,4,stride=2,padding=1)
        self.conv3 = nn.Conv3d(self.num_features*2,self.num_features*4,4,stride=2,padding=1)
        self.conv4 = nn.Conv3d(self.num_features*4,self.num_features*8,4,stride=1) # Last conv has no padding and stride=1

        self.bn1 = nn.BatchNorm3d(self.num_features*2) # First layer doesn't get batch norm
        self.bn2 = nn.BatchNorm3d(self.num_features*4)
        self.bn3 = nn.BatchNorm3d(self.num_features*8)

        self.leaky_relu = nn.LeakyReLU(0.2)

        # TODO: 2 Bottleneck layers
        self.bottleneck = nn.Sequential(
            nn.Linear(self.num_features*8,self.num_features*8),
            nn.ReLU(),
            nn.Linear(self.num_features*8,self.num_features*8),
            nn.ReLU()
        )

        # TODO: 4 Decoder layers
        self.tconv1 = nn.ConvTranspose3d(self.num_features*8*2 ,self.num_features*4,4,stride=1)
        self.tconv2 = nn.ConvTranspose3d(self.num_features*4*2,self.num_features*2,4,stride=2,padding=1)
        self.tconv3 = nn.ConvTranspose3d(self.num_features*2*2,self.num_features,4,stride=2,padding=1)
        self.tconv4 = nn.ConvTranspose3d(self.num_features*2,1,4,stride=2,padding=1)

        self.bn4 = nn.BatchNorm3d(self.num_features*4)
        self.bn5 = nn.BatchNorm3d(self.num_features*2)
        self.bn6 = nn.BatchNorm3d(self.num_features) # No batch norm after last layer. Nor ReLU

        self.relu = nn.ReLU()
    def forward(self, x):
        b = x.shape[0]
        # Encode
        # TODO: Pass x though encoder while keeping the intermediate outputs for the skip connections
        x_e1 = self.leaky_relu(self.conv1(x))
        x_e2 = self.leaky_relu(self.bn1(self.conv2(x_e1)))
        x_e3 = self.leaky_relu(self.bn2(self.conv3(x_e2)))
        x_e4 = self.leaky_relu(self.bn3(self.conv4(x_e3)))

        # Reshape and apply bottleneck layers
        x = x_e4.view(b, -1)
        x = self.bottleneck(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)
        
        # Decode
        # TODO: Pass x through the decoder, applying the skip connections in the process

        
        x = self.relu(self.bn4(self.tconv1(torch.cat((x,x_e4),dim=1)))) # Might need to change the dim later
        x = self.relu(self.bn5(self.tconv2(torch.cat((x,x_e3),dim=1))))
        x = self.relu(self.bn6(self.tconv3(torch.cat((x,x_e2),dim=1))))
        x = self.tconv4(torch.cat((x,x_e1),dim=1))
        
        x = torch.squeeze(x, dim=1) 
        # TODO: Log scaling
        x = torch.abs(x)
        x = torch.log(x+1)
        return x
