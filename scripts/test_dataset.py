import torch
import torch.optim as opt

from training import CombinedLoss

device = torch.device("cuda")
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=5, out_channels=4, init_features=32, pretrained=False).to(device)

optimizer = opt.Adam(model.parameters(), lr=0.001)
class_weights = torch.tensor([0.229, 1.345, 0.796, 16.479]).to(device)
loss_fn = CombinedLoss(class_weights=class_weights, num_classes=4, dice_weight=0.5)

batch_size = 8
x = torch.randn(batch_size, 5, 512, 512).to(device)
y = torch.randint(0, 4, (batch_size, 512, 512)).to(device)

optimizer.zero_grad()
pred = model(x)
loss = loss_fn(pred, y)
loss.backward()
optimizer.step()

print("VRAM used (GB):", torch.cuda.max_memory_allocated() / 1e9)