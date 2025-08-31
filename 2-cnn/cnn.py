'''
Get to 85% + using only CNN (no pretrained models) for CIFAR-10 dataset
'''

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import time
import torch.nn.init as init
import torch.nn.functional as F

torch.set_float32_matmul_precision('high') # uses TF32
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

# -----
# CONFIG
bs = 1024*2
lr = 1e-2
# -----
device = 'cuda'
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip()

])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        # input (bs, 3, 32, 32)
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # self.model.apply(self._init_weights)

    def forward(self, x):
        out = F.elu(self.conv1(x))
        out = self.pool1(out)
        out = F.elu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(out.size(0), -1)

        # FFN
        out = F.elu(self.fc1(out))
        out = self.dropout1(out)
        out = F.elu(self.fc2(out))
        out = self.fc3(out)
        # print(out.shape)

        return out
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)


model = NeuralNet().to(device)
# model.apply(model._init_weights)

model.compile()

loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)

def train_epoch(epoch_ct):
    model.train()
    running_loss = 0
    num_batches = 0

    for step,(x,y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = x.to(device, non_blocking=True), y.to(device) # non-blocking prevents cpu to gpu data transfer blocks(and make code faster)
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        running_loss += loss.item()
        num_batches += 1
        loss.backward()
        optimizer.step()
        # if epoch_ct < 5:
        #     break
    print((running_loss/num_batches))

def test_model():
    model.eval()
    with torch.no_grad():
        correct_pred = 0
        total_pred = 0
        for i,(x,y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device) 
            y_pred = model(x)
            y_pred_class = torch.argmax(y_pred, dim=1)
            correct_pred += (y == y_pred_class).sum().item()
            total_pred += y.shape[0]
        print(f'Train Accuracy:{(correct_pred*100/total_pred):.3f}%')

        correct_pred = 0
        total_pred = 0
        for i,(x,y) in enumerate(test_loader):
            x, y = x.to(device, non_blocking=True), y.to(device) 
            y_pred = model(x)
            y_pred_class = torch.argmax(y_pred, dim=1)
            correct_pred += (y == y_pred_class).sum().item()
            total_pred += y.shape[0]
        print(f'Test Accuracy:{(correct_pred*100/total_pred):.3f}%')
    

start = time.time()
print('Epoch:')
for i in range(100):
    print(i+1,'->', end=' ')
    train_epoch(i)
    if i+1 in [1, 5, 10, 15, 20, 30, 60, 50, 70, 75, 80, 85, 90, 95, 100]:
        print('-----')
        test_model()
        print('-----\n')
print('-----')
end = time.time()
print(f"Training time: {end - start:.2f}s")