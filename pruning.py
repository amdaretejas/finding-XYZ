import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

# Load YOLOv11 model
model = YOLO('yolov11.pt').model  # assume you have a trained model

# Apply pruning to Conv2d layers (structured pruning)
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=0)  # prune 30% channels

# Remove pruning re-parametrization to finalize
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')

# Save pruned model
torch.save(model.state_dict(), 'yolov11_pruned.pt')
