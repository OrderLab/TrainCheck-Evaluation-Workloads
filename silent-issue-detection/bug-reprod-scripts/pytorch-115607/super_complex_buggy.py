# import torch

# def training_loop():
#     input = torch.tensor(
#         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
#     ).reshape(3, 2)

#     model = torch.nn.Sequential(
#         torch.nn.Linear(2, 3),
#         torch.nn.Sigmoid(),
#         torch.nn.Linear(3, 1),
#         torch.nn.Sigmoid(),
#     )

#     params = list(model.parameters())
#     optimizer = torch.optim.Adam(params)

#     for i in range(6):
#         optimizer.zero_grad()
#         # Test that step behaves as expected (a no-op) when grads are set to None
#         if i != 3:
#             output = model(input)
#             loss = output.sum()
#             loss.backward()

#         optimizer.step()
#         print("step", optimizer.state[params[0]]["step"])
#         print("model state", list(model.parameters())[0].data.sum())

# compiled_training_loop = torch._dynamo.optimize("eager", save_config=False)(training_loop)

# print("expected in eager:")
# training_loop()

# print("what actually happens after dynamo:")
# compiled_training_loop()

# xxxx

import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
import json

def img_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def normalize_output(img):
    img = img - img.min()
    img = img / img.max()
    return img

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            total_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    return total_loss, accuracy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load the MNIST dataset
transform = img_transform()
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 30  # Use the same batch size for both training and testing

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = Net().cuda()
criterion = nn.CrossEntropyLoss()
# with open('config.json') as config_file:
#     config = json.load(config_file)
config = {
    "learning_rate": 1e-4,
    "num_epoch": 10,
    "output": "results_origin_lr1e-4"
}

learning_rate = config.get('learning_rate')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epochs = config.get('num_epoch')

# Dictionary to store results
results = {
    'train_loss': [],
    'train_accuracy': [],
    'validation_loss': [],
    'validation_accuracy': []
}
def training_loop():
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.cuda(), label.cuda()
            optimizer.zero_grad()  # Clear previous gradients

            if batch_idx != 50:
                # temporarilly halting the training loop
                pred_score = model(data)
                loss = criterion(pred_score, label)
                loss.backward()

            optimizer.step()

            running_loss += loss.item()
            _, pred = pred_score.topk(1)
            pred = pred.t().squeeze()
            correct += pred.eq(label).sum().item()
            total += label.size(0)

            if batch_idx % 200 == 0:
                print('epoch', epoch, batch_idx, '/', len(train_loader), 'loss', loss.item())

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        print(f'\nTrain set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{total} ({train_accuracy:.0f}%)\n')

        validation_loss, validation_accuracy = evaluate(model, test_loader, criterion, device='cuda')

        # Save results for the current epoch
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        results['validation_loss'].append(validation_loss)
        results['validation_accuracy'].append(validation_accuracy)

    # Save results to JSON
    with open(f"{config.get('output')}.json", 'w') as f:
        json.dump(results, f)

compiled_training_loop = torch._dynamo.optimize("eager", save_config=False)(training_loop)
compiled_training_loop()
# dict_to_save = {
#     'epoch': epochs,
#     'state_dict': model.state_dict(),
#     'optimizer': optimizer.state_dict(),
# }
# ckpt_file = 'a.pth.tar'
# save_checkpoint(dict_to_save, ckpt_file)
# print('save to ckpt_file', ckpt_file)
