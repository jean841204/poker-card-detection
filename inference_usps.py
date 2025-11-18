### https://docs.ultralytics.com/modes/predict/#inference-arguments

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO(r'D:\Jean\01_modeltraining\ultralytics_yolo\runs\classify\usps\weights\best.pt')
# print(model.names)

# Run inference on 'bus.jpg' with arguments
model.predict(source=r'D:\Jean\02_project\usps\inverted_images', 
              save=True,
              save_txt=True, 
              show_boxes=True,
              imgsz=32,
              project = 'predict',
              name='jean_write',
              conf=0.3)