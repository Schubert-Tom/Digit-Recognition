import torch.nn as nn
import torch.nn.functional as F
# Simple functional stuff

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        # Layers
        # Lineaer Transformation y=xAT+b --> Skalarprodukt aller Gewichte mit allen Inputs
        # Output per neuron scalar --> vector of out_feature scalars
        self.lin1=nn.Linear(784,128)
        self.lin2=nn.Linear(128,64)
        self.lin3=nn.Linear(64,10)
        # Relu if<0-->0 if<0 x
        # Linear Activation function
        self.relu=nn.ReLU()
        
    def forward(self,x):
        # Look at the Architecture inside the notebook
        x=self.lin1(x)
        x=self.relu(x)
        x=self.lin2(x)
        x=self.relu(x)
        x=self.lin3(x)
        # Logsoftmax for classification
        return F.log_softmax(x,dim=1)
