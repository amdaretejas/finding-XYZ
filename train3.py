from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s-obb.yaml")  # build a new model from YAML
model = YOLO("yolo11s-obb.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11s-obb.yaml", task="detect").load("yolo11s.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(
    data="data.yaml",     
    epochs=100,           
    imgsz=640,            
    batch=8,              
    lr0=0.01,             
    lrf=0.01,             
    patience=5,           
    optimizer="Adam",     
    weight_decay=0.0005,
    hsv_h=0.015, 
    hsv_s=0.7, 
    hsv_v=0.4, 
    degrees=10, 
    translate=0.1, 
    scale=0.5, 
    shear=2.0,
    perspective=0.0001,
    flipud=0.5, 
    fliplr=0.5,  
    save=True,      
    project="result", exist_ok=False,
    pretrained = True,
    single_cls = True,
    multi_scale=True,
    cos_lr=True,
    resume=True,
    box=10,
    pose=20,
    plots=True,
    warmup_momentum=0.8,
    warmup_epochs= 3.0,
    momentum= 0.937
)