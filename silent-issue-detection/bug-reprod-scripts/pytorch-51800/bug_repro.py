import torch
if torch.cuda.is_available():
    print("Using the GPU. You are good to go!")
    device = 'cuda'
else:
    print("Using the CPU. Overall speed may be slowed down")
    device = 'cpu'

from torch import nn
import torch


class SpectralNormTestModel(nn.Module):
    # refering to https://github.com/pytorch/pytorch/issues/51800
        def __init__(self):
            super(SpectralNormTestModel, self).__init__()
            feature_count = 1000

            def init_(layer: nn.Linear):
                nn.init.zeros_(layer.bias.data)
                nn.init.orthogonal_(layer.weight.data, 1)

            self.fc1 = nn.Linear(feature_count, feature_count)
            init_(self.fc1)
            self.fc1 = nn.utils.spectral_norm(self.fc1)
            self.act1 = nn.PReLU(feature_count)

            self.fc2 = nn.Linear(feature_count, feature_count)
            init_(self.fc2)
            self.fc2 = nn.utils.spectral_norm(self.fc2)
            self.act2 = nn.PReLU(feature_count)

            self.out = nn.Linear(feature_count, 1)
            init_(self.out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.out(self.act2(self.fc2(self.act1(self.fc1(x)))))
        

x = torch.randn(10000, 1000, device='cuda')
model = SpectralNormTestModel().cuda()
model = model.eval()
y = model(x)
print(f"Mean absolute output of model directly evaluated in the eval mode: {y.abs().mean()}\n")
print("You can see a large value and no initialization happens here.\n")

print("All layers parameter:\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Value: {param}")

print("All registered buffer:\n")
for name, buffer in model.named_buffers():
    print(f"Buffer: {name} | Size: {buffer.size()} | Value: {buffer}")


print("Set model to train mode.\n")
model = model.train()
z = model.forward(x)
print("Do a dumy forward.\n")
y = model.eval()(x)
print("Set model back to eval mode.\n")

print(f"Mean absolute output of after a dumy forward is: {y.abs().mean()}")
print("You can see a small value and initialization happened.\n")

print("All layers parameter:\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Value: {param}")

print("All registered buffer:\n")
for name, buffer in model.named_buffers():
    print(f"Buffer: {name} | Size: {buffer.size()} | Value: {buffer}")

