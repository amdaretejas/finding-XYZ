import torch
from ultralytics import YOLO

model = YOLO('yolov11.pt').model
model.eval()

# Convert to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save('yolov11_fp32.pt')

# Dynamic quantization (CPU optimized)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(quantized_model.state_dict(), 'yolov11_int8.pt')

# yolo export model=yolov11.pt format=tflite int8
