####### V1.2 of the model training #####
#### This script is writen to run on a Kaggle notebook with GPU ###

from ultralytics import YOLO
import locale

# Load a model
model = YOLO("yolo11m.pt")
#model = YOLO("/kaggle/working/runs/detect/train14/weights/best.pt")
# Train the model with data augmentation parameters
train_results = model.train(
    data="/kaggle/input/zebra-2800/dataset/zebra.yaml",  # path to dataset YAML
    epochs=350,  # number of training epochs
    imgsz=640,  # training image size
    device="0,1",  # device to run on, i.e. device=0 or device=cpu
    batch=32,
    patience=50,
    weight_decay=0.0005,
    lr0=0.008, # Légèrement inférieur au défaut (~0.01)
    lrf=0.1    # Le LR final sera lr0 * lrf. Défaut est souvent 0.01    
)
/kaggle/input/
# Evaluate model performance on the validation set
metrics = model.val()
