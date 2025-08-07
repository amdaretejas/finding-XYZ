from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s-obb.yaml")  # build a new model from YAML
model = YOLO("yolo11s-obb.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11s-obb.yaml").load("yolo11s.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(
    data="data.yaml",     # Path to your dataset config file
    epochs=100,            # More epochs with early stopping in mind
    imgsz=640,            # Input image size
    batch=8,              # Small batch due to limited data
    lr0=1e-3,             # Initial learning rate
    lrf=0.01,             # Final learning rate fraction
    patience=5,           # Early stopping patience
    optimizer="Adam",     # Adam performs better on small datasets
    weight_decay=1e-4,    # Helps with regularization
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # Color augmentation
    degrees=10, translate=0.1, scale=0.5, shear=2.0,  # Geometric augmentation
    perspective=0.0001,   # Slight distortion for generalization
    flipud=0.5, fliplr=0.5,  # Flipping images
    save=True,            # Save model checkpoints
    project="result", exist_ok=False,
    pretrained = True,
    single_cls = True,
    multi_scale=True,
    cos_lr=True,
    resume=True,
    box=10,
    pose=20,
    plots=True
)