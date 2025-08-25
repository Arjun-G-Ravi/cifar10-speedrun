import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import time

torch.set_float32_matmul_precision('high') # uses TF32

# CONFIG
bs = 64
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True) # pin_memory loads thing in RAM(making cpu-to-gpu transfer fast), drop_last is to prevent shape issue at last datapoint

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=2, drop_last=True) # 


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3*32*32, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 64),
        )

    def forward(self, x):
        x = x.view(bs, -1)
        logits = self.model(x)
        return logits

model = NeuralNet().to('cuda')
model.compile()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_epoch():
    model.train()
    avg_loss = 0

    for i,(x,y) in enumerate(train_loader):
        x, y = x.flatten().to('cuda', non_blocking=True), y.to('cuda') # non-blocking prevents cpu to gpu data transfer blocks(and make code faster)
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        avg_loss += loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print((avg_loss/bs).item())

start = time.time()
for i in range(5):
    l = train_epoch()
end = time.time()
print(f"Time taken for training: {end - start} seconds")
