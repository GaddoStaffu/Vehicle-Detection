import os
import urllib.request
from ultralytics import YOLO

def download_model(model_filename, model_url):
    model_path = os.path.join(os.getcwd(), model_filename)
    if not os.path.exists(model_path):
        print(f"Downloading {model_filename}...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print(f"Model downloaded successfully and saved as {model_path}")
        except Exception as e:
            print(f"Error downloading the model: {e}")
    else:
        print(f"Model already exists at {model_path}")
    return model_path

def train_model(model_path, data_path, resume=False):
    model = YOLO(model_path)  # Load the best.pt checkpoint
    model.train(
        data=data_path,
        epochs=1000,  # Set the new total number of epochs
        imgsz=640,
        batch=8,
        device=0,
        patience=50,
        amp=True,
        workers=4,
        optimizer='AdamW',
        cache='disk',
        dropout=0.1,
        resume=resume,  # Resume training from the checkpoint
        project=r"outputs",  # Save outputs to D: drive
        name="vehicle_detection",  # Name of the training run
        exist_ok=False  # Overwrite existing directory if it exists
    )
    print("Training complete! Model is now trained for detecting only bikes and cars.")

if __name__ == '__main__':
    # Point to the best.pt checkpoint
    model_filename = os.path.join("yolov8s.pt")
    model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt"
    model_path = download_model(model_filename, model_url)
    data_path = os.path.join(os.getcwd(), "dataset", "data.yaml")
    
    # Continue training from the best.pt checkpoint
    train_model(model_path, data_path, resume=False)