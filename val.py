from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s-obb.pt")  # load an official model
model = YOLO("result/train3/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val(data="data.yaml")  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps  # a list contains map50-95(B) of each category