# All functions that can be optimized with params
import torch.nn as nn
# Simple functional stuff
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128,
                kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.tns1 = nn.Conv2d(in_channels=128, out_channels=4,
                kernel_size=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16,
                kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16,
                kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32,
                kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=32)
                
        self.pool2 = nn.MaxPool2d(2, 2)
        self.tns2 = nn.Conv2d(in_channels=32, out_channels=16,
                kernel_size=1, padding=1)
                
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16,
                kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=16)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=32,
                kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=32)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=10,
                kernel_size=1, padding=1)
        self.gpool = nn.AvgPool2d(kernel_size=7)
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
            x = self.tns1(self.drop(self.bn1(F.relu(self.conv1(x)))))
            x = self.drop(self.bn2(F.relu(self.conv2(x))))
            x = self.pool1(x)
            x = self.drop(self.bn3(F.relu(self.conv3(x))))
            x = self.drop(self.bn4(F.relu(self.conv4(x))))
            x = self.tns2(self.pool2(x))
            x = self.drop(self.bn5(F.relu(self.conv5(x))))
            x = self.drop(self.bn6(F.relu(self.conv6(x))))
            # outputs out_channels=10 -->10 numbers
            x = self.conv7(x)
            # 
            x = self.gpool(x)
            # same as tensor.view == np.reshape
            #
            #  [[1, 2, 3],
            #   [4, 5, 6]]
            # -->
            #    
            #[[ 1,  4],
            # [ 2,  5],
            # [ 3,  6]]
            #
            #
            x = x.view(-1, 10)
            # Log Softmax classificator for using probability to chose correct class for input vector
            # normal softmax: https://www.youtube.com/watch?v=EuZZ6plg2Tk&ab_channel=BhaveshBhatt
            return F.log_softmax(x,dim=1)



### Reuse a Network @ https://nextjournal.com/gkoehler/pytorch-mnist

#continued_network = Net()
#continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,
#                                momentum=momentum)


#network_state_dict = torch.load()
#continued_network.load_state_dict(network_state_dict)

#optimizer_state_dict = torch.load()
#continued_optimizer.load_state_dict(optimizer_state_dict)