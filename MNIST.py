import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# downloading data
train = datasets.MNIST("", train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))

# creating train and test set
trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)

# observing first batch
for data in trainset:
    print(data)
    break
    pass

x, y = data[0][0], data[1][0]
print(y)

# plotting value
plt.imshow(x.view(28,28))


# viewing counts of each digit
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
    Xs, ys = data
    for y in ys:
        counter_dict[int(y)] +=1
    pass

print(counter_dict)

# creating class
class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        # creating 4 fully connected layers
        self.fcl1 = nn.Linear(28 * 28, 64)
        self.fcl2 = nn.Linear(64, 64)
        self.fcl3 = nn.Linear(64, 64)
        self.fcl4 = nn.Linear(64, 10)
    
    def forward(self, data):
        # running 3 rectified linear activation funtions 
        data = F.relu(self.fcl1(data))
        data = F.relu(self.fcl2(data))
        data = F.relu(self.fcl3(data))
        data = self.fcl4(data)
        return F.log_softmax(data, dim = 1)

# creating an example to see if NN was able to take in data and return output
example = torch.rand(28,28).view(1, 28*28)
model = Net()
output = model(example)
print(output)

optimizer = optim.Adam(model.parameters(), lr = 0.001)
epoch = 3

for e in range(epoch):
    for data in trainset:
        X, y = data
        # if you dont add zero gradient, the values keep getting added, 
        model.zero_grad()
        output = model(X.view(-1, 28*28))
        # applying loss function 
        loss = F.nll_loss(output, y)
        # applying backward propegation 
        loss.backward()
        # adjusts weights
        optimizer.step()
    print(loss)
    
correct = 0
total = 0

# going over the testing set, therefore we do not want to caclulate gradients
with torch.no_grad():
    for data in trainset:
        X, y = data
        output = model(X.view(-1, 28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
print("Accuracy:", round(correct/total,3))

plt.imshow(X[0].view(28, 28))
print(torch.argmax(model(X[0].view(-1, 28*28))[0]))
