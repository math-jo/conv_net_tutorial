import torch
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, Linear, BatchNorm2d, BatchNorm1d, ReLU, Flatten
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self):
        """
        This function initializes the Net class and defines the network architecture:

        Args:

        Returns:
        """

        super(Net,self).__init__()

        # self.conv_net = torch.nn.Sequential(
        #     Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2),
        #     ReLU(),
        #     MaxPool2d(2,2),
        #     Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2),
        #     ReLU(),
        #     MaxPool2d(2,2),
        #     Flatten(start_dim=1),
        #     Linear(in_features= 16*5*5, out_features=120),
        #     ReLU(),
        #     Linear(in_features=120, out_features=84),
        #     ReLU(),
        #     Linear(in_features=84, out_features=10),
        # )

        # self.conv_net = torch.nn.Sequential(
        #     Conv2d(3, 6, 5),
        #     ReLU(),
        #     MaxPool2d(2,2),
        #     Conv2d(6, 16, 5),
        #     ReLU(),
        #     MaxPool2d(2,2),
        #     Flatten(start_dim=1),
        #     Linear(16 * 5 * 5, 120),
        #     ReLU(),
        #     Linear(120, 84),
        #     ReLU(),
        #     Linear(84, 10),
        # )

        self.conv_net = torch.nn.Sequential(
            Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2),
            BatchNorm2d(6),
            ReLU(),
            MaxPool2d(2,2),
            Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(2,2),
            Flatten(start_dim=1),
        )
            
        self.lin_net = torch.nn.Sequential(
            
            Linear(in_features=1024, out_features=2000),
            ReLU(),
            Linear(in_features=2000, out_features=10),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function receives the input x and pass it over the network, returning the model outputs:

        Args:
            - x (tensor): input data

        Returns:
            - out (tensor): network output given the input x
        """
        out = self.conv_net(x)
        #print(out.shape)
        out = self.lin_net(out)
        


        return out