from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("result/train2/weights/best.pt")

# Define search space
search_space = {
    "momentum": (0.6, 0.98),
    "lr0": (1e-5, 1e-1),
    "lrf": (0.01, 1.0),
    "weight_decay": (0.0, 0.001),
    "warmup_epochs": (0, 5),
    "warmup_momentum": (0.0, 0.95),
    "box": (0.02, 0.2),
    "cls": (0.02, 0.2),
    "hsv_h": (0.0, 0.1),
    "hsv_s": (0.0, 0.9),
    "hsv_v": (0.0, 0.9),
    "degrees": (0.0, 45.0),
    "translate": (0.0, 0.9),
    "scale": (0.0, 0.9),
    "shear": (0.0, 10.0),
    "perspective": (0.0, 0.001),
    "flipud": (0.0, 1.0),
    "fliplr": (0.0, 1.0),
    "mosaic": (0.0, 1.0),
    "mixup": (0.0, 1.0),
    "copy_paste": (0.0, 1.0)
}

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(
    data="data.yaml",
    epochs=30,
    iterations=30,
    optimizer="AdamW",
    space=search_space,
    plots=True,
    save=True,
    val=True,
)


# # Resume previous run
# results = model.tune(data="coco8.yaml", epochs=50, iterations=300, space=search_space, resume=True)

# # Resume tuning run with name 'tune_exp'
# results = model.tune(data="coco8.yaml", epochs=50, iterations=300, space=search_space, name="tune_exp", resume=True)