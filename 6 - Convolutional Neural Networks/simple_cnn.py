import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms

device = 'cuda'

class SimpleCnn(nn.Module):
    def __init__(self ,in1 = 1, out1 = 32, 
                 in2 = 32, out2= 64, 
                 in3 = 64, out3 = 64,
                 in4 = 64, out4 = 32, 
                 kersize = 3, striding = 1, pad = 1):
        super(SimpleCnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= in1, out_channels = out1, kernel_size= kersize, stride = striding, padding = pad).to(device)
        self.conv2 = nn.Conv2d(in_channels= in2, out_channels = out2, kernel_size= kersize, stride = striding, padding = pad).to(device)
        self.conv3 = nn.Conv2d(in_channels= in3, out_channels = out3, kernel_size= kersize, stride = striding, padding = pad).to(device)
        self.conv4 = nn.Conv2d(in_channels= in4, out_channels = out4, kernel_size= kersize, stride = striding, padding = pad).to(device)

        self.bn1 = nn.BatchNorm2d(num_features=out1, device=device)
        self.bn2 = nn.BatchNorm2d(num_features=out2, device=device)
        self.bn3 = nn.BatchNorm2d(num_features=out3, device=device)
        self.bn4 = nn.BatchNorm2d(num_features=out4, device=device)
        self.bn5 = nn.BatchNorm1d(num_features=128, device=device)
        
        self.do1 = nn.Dropout2d(p = 0.2)
        self.do2 = nn.Dropout2d(p = 0.2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0).to(device)
        self.hl1 = nn.Linear(32*7*7,128).to(device)
        self.hl2 = nn.Linear(128,10).to(device)
    
    def forward(self, x):
        x = self.do1(x).to(device)
        x = F.relu(self.conv1(x)).to(device)
        x = self.bn1(x).to(device)
        x = self.pool(self.bn2(F.relu(self.conv2(x)))).to(device)
        x = self.do2(x).to(device)

        

        x = F.relu(self.conv3(x)).to(device)
        x = self.bn3(x).to(device)
        x = self.pool(self.bn4(F.relu(self.conv4(x)))).to(device)
        x = self.do2(x).to(device)

        x = x.view(-1,32*7*7)

        x = self.bn5(F.relu(self.hl1(x))).to(device)
        x = self.do2(x).to(device)
        x = self.hl2(x).to(device)

        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalizing the data
])

train_data = datasets.FashionMNIST(root="./data", train=True, download=True,transform=transform)
test_data = datasets.FashionMNIST(root="./data", train=False, download=True,transform=transform)

train_data_download = torch.utils.data.DataLoader(train_data, batch_size= 64, shuffle= True)
test_data_download = torch.utils.data.DataLoader(test_data, batch_size= 64, shuffle= False)

"""
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, 0.0, 0.001)  # Initialize weights between 0 and 0.001
        if m.bias is not None:
            nn.init.uniform_(m.bias, 0.0, 0.001)  # Initialize biases between 0 and 0.001

"""
model = SimpleCnn().to(device)
#model.apply(initialize_weights)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.005)


num_epochs = 10
for epoch in range(num_epochs):
    total_loss_in_one_epoch = 0.0
    for i,(image, label) in enumerate(train_data_download):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label)
        
        loss.backward()
        optimizer.step()

        total_loss_in_one_epoch += loss.item()

        if (i+1) % 99 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_data_download)}]: {total_loss_in_one_epoch}')

model.eval() # only has affect on certain modules of nn class, like Dropout, BatchNorm etc.
loss_test = 0.0
total = 0
with torch.no_grad():
    
    for image, label in test_data_download:
        image, label = image.to(device), label.to(device)
        _ , predictions = torch.max(model(image),1)
        total += label.size(0) # returns the number of rows in each batch
        loss_test += (predictions == label).sum().item()

print(f'Accuracy: {(100*loss_test)/total}%')

# with lr = 0.01
    # Accuracy: 88.28% 
    # in1 = 1, out1 = 32, in2 = 32, out2= 64, in3 = 64, out3 = 64, in4 = 64, out4 = 32,  kersize = 3, striding = 1, pad = 1

# with lr = 0.001:
    # Accuracy: 92.47%              
    # Accuracy: 92.32%
    # in1 = 1, out1 = 32, in2 = 32, out2= 64, in3 = 64, out3 = 64, in4 = 64, out4 = 32,  kersize = 3, striding = 1, pad = 1

# BATCH NORMALIZATION with lr = 0.001 after ReLU layer:
    # Accuracy: 92.82% 
    # Accuracy: 92.96%
    # Accuracy: 92.75%
    # + bn5 after final relu Accuracy: 92.57%
    # lr = 0.005 & +bn5 Accuracy: 91.92%
    # in1 = 1, out1 = 32, in2 = 32, out2= 64, in3 = 64, out3 = 64, in4 = 64, out4 = 32,  kersize = 3, striding = 1, pad = 1

# DROPOUT with lr = 0.001 
    # in1 = 1, out1 = 32, in2 = 32, out2= 64, in3 = 64, out3 = 64, in4 = 64, out4 = 32,  kersize = 3, striding = 1, pad = 1

# BATCH NORMALIZATION && DROPOUT input & normal layers with lr = 0.01 (slower than normal, with 15 epochs)
    # with lr = 0.01, with 15 epochs, p=0.02 Accuracy: 91.67%
    # with lr = 0.01, with 10 epochs, p=0.01 Accuracy: 91.39%
    # with lr = 0.005, with 10 epochs, p=0.02, + bn5 after final relu Accuracy: 93.29%
    # with lr = 0.005, with 10 epochs, p=0.02, + bn5 before final relu Accuracy: 93.25%
    # with lr = 0.007, with 10 epochs, p=0.02, + bn5 before final relu Accuracy: 91.9%
    # with lr = 0.005, with 10 epochs, p=0.03, + bn5 before final relu Accuracy: 92.16%
    # in1 = 1, out1 = 32, in2 = 32, out2= 64, in3 = 64, out3 = 64, in4 = 64, out4 = 32,  kersize = 3, striding = 1, pad = 1
