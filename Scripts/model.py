from lib import *

##################### MODEL DEFINITION #####################


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # (2, 40, 1376) -> (8,20,688)
        self.cnn1 = nn.Conv2d(2, 8, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        self.relu1 = nn.ReLU() 
        self.bn1 = nn.BatchNorm2d(8)
        self.maxpool1 = nn.MaxPool2d(kernel_size=[2,2], padding=[0,0])
#         self.dropout1 = nn.modules.Dropout2d(p=0.3)
        
        # (8,20,688) -> (16,10,344)
        self.cnn2 = nn.Conv2d(8, 16, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=[2,2], padding=[0,0])
#         self.dropout2 = nn.modules.Dropout2d(p=0.2)
        
        # (16,10,344) -> (32,5,172)
        self.cnn3 = nn.Conv2d(16, 32, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool3 = nn.MaxPool2d(kernel_size=[2,2], padding=[0,0])
#         self.dropout3 = nn.modules.Dropout2d(p=0.2)
        
        # (32,5,172) -> (64,5,86)
        self.cnn4 = nn.Conv2d(32, 64, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        self.maxpool4 = nn.MaxPool2d(kernel_size=[1,2], padding=[0,0])
#         self.dropout4 = nn.modules.Dropout2d(p=0.2)
        
        # (64,5,86) -> (64,5,43)
        self.cnn5 = nn.Conv2d(64, 64, kernel_size=[3,3], stride=[1,1], padding=[1,1])
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(64)
        self.maxpool5 = nn.MaxPool2d(kernel_size=[1,2], padding=[0,0])
#         self.dropout5 = nn.modules.Dropout2d(p=0.2)
    
        self.flatten = nn.Flatten()
        
        self.linear1 = nn.Linear(64*5*43, 1024)
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm1d(1024)
        self.droupout6 = nn.modules.Dropout(p=0.3)
        
        self.linear2 = nn.Linear(1024, 128)
        self.relu7 = nn.ReLU()
        self.bn7 = nn.BatchNorm1d(128)
        self.droupout7 = nn.modules.Dropout(p=0.3)
        
        self.linear3 = nn.Linear(128, 5)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
#         print(x.shape)
        
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.maxpool1(out)
#         out = self.dropout1(out)
#         print(out.shape)
        
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.maxpool2(out)
#         out = self.dropout2(out)
#         print(out.shape)
        
        out = self.cnn3(out)
        out = self.relu3(out)
        out = self.bn3(out)
        out = self.maxpool3(out)
#         out = self.dropout3(out)
#         print(out.shape)

        out = self.cnn4(out)
        out = self.relu4(out)
        out = self.bn4(out)
        out = self.maxpool4(out)
#         out = self.dropout4(out)
#         print(out.shape)

        out = self.cnn5(out)
        out = self.relu5(out)
        out = self.bn5(out)
        out = self.maxpool5(out)
#         out = self.dropout5(out)
#         print(out.shape)
        
        out = self.flatten(out)
#         print(out.shape)
        
        out = self.linear1(out)
        out = self.relu6(out)
        out = self.bn6(out)
        out = self.droupout6(out)
#         print(out.shape)
    
        out = self.linear2(out)
        out = self.relu7(out)
        out = self.bn7(out)
        out = self.droupout7(out)
#         print(out.shape)
        
        out = self.linear3(out)
        out = self.softmax(out)
#         print(out.shape)
        
        return out