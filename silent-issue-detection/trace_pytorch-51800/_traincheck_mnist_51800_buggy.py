import torch
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "/home/yuxuan/gitrepos/traincheck-ae-resources/silent-issue-detection/bug-reprod-scripts/pytorch-51800/trace_pytorch-51800"

from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)

import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = False
import traincheck.proxy_wrapper.proxy_config as proxy_config
proxy_config.__dict__.update({'proxy_log_dir': '/home/yuxuan/gitrepos/traincheck-ae-resources/silent-issue-detection/bug-reprod-scripts/pytorch-51800/trace_pytorch-51800/proxy_log.json'})

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
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.nn as nn
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(nn, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.optim as optim
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(optim, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(DataLoader, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()
import torch.nn.utils.spectral_norm
from traincheck.instrumentor.tracer import Instrumentor
Instrumentor(torch.nn.utils.spectral_norm, scan_proxy_in_args=True, use_full_instr=False, funcs_to_instr=None, API_dump_stack_trace=False).instrument()

class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.utils.spectral_norm(self.conv2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc1 = nn.utils.spectral_norm(self.fc1)
        self.fc2 = nn.Linear(128, 10)
        self.fc2 = nn.utils.spectral_norm(self.fc2)

    def forward(self, x):
        x = torch.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
model = SimpleCNN()
model = Proxy(model, recurse=True, logdir=proxy_config.proxy_log_dir, var_name='model')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for (batch_idx, (data, target)) in enumerate(train_loader):
        (data, target) = (data.to(device), target.to(device))
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        if batch_idx == 10:
            break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (data, target) in test_loader:
            (data, target) = (data.to(device), target.to(device))
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.0f}%)\n')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(1, 3):
        test(model, device, test_loader)
        train(model, device, train_loader, optimizer, epoch)
if __name__ == '__main__':
    main()