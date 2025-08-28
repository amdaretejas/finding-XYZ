from ultralytics import YOLO

# Load your previously fine-tuned model
model = YOLO("result/train2/weights/best.pt")

# Train the model on the new dataset
results = model.train(
    data="data.yaml",        # Your dataset config
    epochs=100,              # Max epochs
    imgsz=640,               # Input size
    batch=16,                # Increase if GPU can handle it
    lr0=0.001,               # Initial LR
    lrf=0.01,                # Final LR fraction
    patience=20,             # Early stopping patience (increase for large dataset)
    optimizer="AdamW",       # AdamW for better regularization
    weight_decay=0.0005,     # Slightly higher for regularization
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # Color augmentation
    degrees=10, translate=0.1, scale=0.5, shear=0.1,  # Geometric augmentation (reduced for stability)
    perspective=0.0005,      # Slight perspective change
    flipud=0.5, fliplr=0.5,  # Flipping
    mosaic= 1.0,
    mixup=0.2,               # MixUp augmentation for regularization
    copy_paste=0.1,          # Copy-Paste augmentation
    save=True,               # Save checkpoints
    project="result2",  # New project folder
    exist_ok=False,          # Avoid overwrite
    pretrained=False,        # Not needed since we load best.pt
    single_cls=True,         # Keep only if you have ONE class
    multi_scale=False,       # Keep it off for OBB stability
    cos_lr=True,             # Cosine LR schedule
    resume=False,            # Start fresh from best.pt weights
    plots=True               # Generate plots
)
