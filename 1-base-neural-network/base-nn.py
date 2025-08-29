import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import time
import torch.nn.init as init

torch.set_float32_matmul_precision('high') # uses TF32
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# -----
# CONFIG
bs = 1024 * 2
lr = 1e-3
# -----
device = 'cuda'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), # these are  mean and std of cifar10 dataset over 3 color channels(this have massive role)
    transforms.RandomHorizontalFlip(),

])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True) # pin_memory loads thing in RAM(making cpu-to-gpu transfer fast), drop_last is to prevent shape issue at last datapoint

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True) # 


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3*32*32, 1024),
            nn.ELU(),
            nn.Dropout(.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(.4),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(.1),
            nn.Linear(128, 10),
        )
        self.model.apply(self._init_weights)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        logits = self.model(x)
        return logits
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)


model = NeuralNet().to(device)
model.compile()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

def train_epoch():
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
for i in range(30):
    print(i+1,'->', end=' ')
    train_epoch()
    if (i+1)%10 == 0 and i > 1:
        print('-----')
        test_model()
        print('-----\n')
print('-----')
end = time.time()
print(f"Training time: {end - start:.2f}s")