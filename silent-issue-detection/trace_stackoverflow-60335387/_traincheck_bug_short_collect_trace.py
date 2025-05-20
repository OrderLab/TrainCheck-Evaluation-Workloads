import matplotlib
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/home/yuxuan/gitrepos/traincheck-ae-resources/silent-issue-detection/bug-reprod-scripts/stackoverflow-60335387/traincheck_run_bug_short_collect_trace_torch_2.2.2+cu121_2025-05-09_00-20-08"

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
import traincheck.proxy_wrapper.proxy_config as proxy_config
proxy_config.__dict__.update({'proxy_log_dir': '/home/yuxuan/gitrepos/traincheck-ae-resources/silent-issue-detection/bug-reprod-scripts/stackoverflow-60335387/traincheck_run_bug_short_collect_trace_torch_2.2.2+cu121_2025-05-09_00-20-08/proxy_log.json'})

from traincheck.proxy_wrapper.proxy import Proxy

import glob
import importlib
from traincheck.proxy_wrapper.proxy_config import auto_observer_config
spec = importlib.util.find_spec('traincheck')
if spec and spec.origin:
    traincheck_folder = os.path.dirname(spec.origin)
    print("traincheck folder: ", traincheck_folder)
else:
    raise Exception("traincheck is not installed properly")
print("auto observer enabled with observing depth: ", auto_observer_config["enable_auto_observer_depth"])
enable_auto_observer_depth = auto_observer_config["enable_auto_observer_depth"]
neglect_hidden_func = auto_observer_config["neglect_hidden_func"]
neglect_hidden_module = auto_observer_config["neglect_hidden_module"]
observe_then_unproxy = auto_observer_config["observe_then_unproxy"]
observe_up_to_depth = auto_observer_config["observe_up_to_depth"]
if observe_up_to_depth:
    print("observe up to the depth of the function call")
else:
    print("observe only the function call at the depth")
from traincheck.static_analyzer.graph_generator.call_graph_parser import add_observer_given_call_graph

log_files = glob.glob(
    os.path.join(traincheck_folder, "static_analyzer", "func_level", "*.log")
)
print("log_files: ", log_files)
for log_file in log_files:
    add_observer_given_call_graph(
        log_file,
        depth=enable_auto_observer_depth,
        observe_up_to_depth=observe_up_to_depth,
        neglect_hidden_func=neglect_hidden_func,
        neglect_hidden_module=neglect_hidden_module,
        observe_then_unproxy=observe_then_unproxy,
    )
matplotlib.use('Agg')
import torch
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.nn as nn
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(nn, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.optim as optim
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(optim, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.nn.functional as F
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(F, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from torch.utils.data import DataLoader
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(DataLoader, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
import json

def img_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

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
        return x

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        batch_idx = 0
        for (data, target) in data_loader:
            batch_idx += 1
            (data, target) = (data.to(device, non_blocking=True), target.to(device, non_blocking=True))
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            if batch_idx == 10:
                break
    total_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)
    return (total_loss, accuracy)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
transform = img_transform()
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
model = Net().cuda()
model = Proxy(model, recurse=True, logdir=proxy_config.proxy_log_dir, var_name='model')
criterion = nn.CrossEntropyLoss()
config = {'learning_rate': 0.0001, 'is_fixed': False, 'num_epoch': 1, 'output': 'results_origin_lr1e-4'}
learning_rate = config.get('learning_rate')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epochs = config.get('num_epoch')
results = {'train_loss': [], 'train_accuracy': [], 'validation_loss': [], 'validation_accuracy': []}
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for (batch_idx, (data, label)) in enumerate(train_loader):
        (data, label) = (data.cuda(), label.cuda())
        if config.get('is_fixed'):
            optimizer.zero_grad()
        pred_score = model(data)
        loss = criterion(pred_score, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        (_, pred) = pred_score.topk(1)
        pred = pred.t().squeeze()
        correct += pred.eq(label).sum().item()
        total += label.size(0)
        if batch_idx == 50:
            break
        print(batch_idx)
        if batch_idx % 200 == 0:
            print('epoch', epoch, batch_idx, '/', len(train_loader), 'loss', loss.item())
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100.0 * correct / total
    print(f'\nTrain set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{total} ({train_accuracy:.0f}%)\n')
    (validation_loss, validation_accuracy) = evaluate(model, test_loader, criterion, device='cuda')
    results['train_loss'].append(train_loss)
    results['train_accuracy'].append(train_accuracy)
    results['validation_loss'].append(validation_loss)
    results['validation_accuracy'].append(validation_accuracy)
with open(f"{config.get('output')}.json", 'w') as f:
    json.dump(results, f)
dict_to_save = {'epoch': epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
ckpt_file = 'a.pth.tar'
save_checkpoint(dict_to_save, ckpt_file)
print('save to ckpt_file', ckpt_file)