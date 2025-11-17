from ultralytics import YOLO

# Load your previously trained model
model = YOLO("result4/train/weights/best.pt")  # This is your fine-tuned model

# Fine-tune the model further on the updated dataset
results = model.train(
    data="data.yaml",          # Dataset config
    epochs=200,                # Increase to allow longer training
    imgsz=1280,                 # Input image size
    batch=8,                  # Adjust per GPU capacity
    lr0=0.00025,                # Lower LR for fine-tuning
    lrf=0.01,                  # Final LR fraction
    patience=50,               # Longer patience for early stopping
    optimizer="AdamW",         # Optimizer with weight decay support
    weight_decay=0.0001,       # Regularization
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # Color augmentations
    degrees=20, translate=0.2, scale=0.8, shear=0.2,  # Stronger geometric aug.
    perspective=0.001,         # Slight increase for OBB robustness
    flipud=0.5, fliplr=0.5,    # Flipping
    mosaic=1.0,                # Keep Mosaic
    mixup=0.2,                 # MixUp augmentation
    copy_paste=0.1,            # Copy-Paste augmentation
    box=10,
    pose=20,
    save=True,                 # Save checkpoints
    project="result4",         # Use a new folder for this run
    exist_ok=False,            # Don't overwrite
    pretrained=True,          # Already loaded from best.pt
    single_cls=True,           # Set to False if you have multiple subtypes
    multi_scale=False,         # Keep off for OBB
    cos_lr=True,               # Cosine LR scheduling
    resume=True,              # Important: don't resume, since you're fine-tuning
    plots=True,                # Save training plots
)
