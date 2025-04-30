import cv2
import os
from tkinter import Tk, Button, Label, filedialog, messagebox
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO(r"outputs\yolov8s_first_model\weights\best.pt")  # Replace with the path to your trained model

def process_frame(frame):
    # Define a color mapping for each class
    class_colors = {
        "car": (0, 255, 0),       # Green
        "bike": (255, 0, 0),      # Blue
    }

    # Define a label replacement mapping
    label_replacements = {
        "bike": "motor",  # Replace "bike" with "motor"
    }

    results = model(frame)
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            label = model.names[int(cls)]
            
            # Replace the label if a replacement is defined
            label = label_replacements.get(label, label)
            
            # Get the color for the class, default to white if not defined
            color = class_colors.get(model.names[int(cls)], (255, 255, 255))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def process_image(image_path):
    frame = cv2.imread(image_path)
    frame = process_frame(frame)
    cv2.imshow("YOLO Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    capture = cv2.VideoCapture(video_path)
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        frame = process_frame(frame)
        cv2.imshow("YOLO Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

def process_webcam():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        messagebox.showerror("Error", "Unable to access the webcam.")
        return
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = process_frame(frame)
        cv2.imshow("YOLO Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        process_video(file_path)

def start_webcam():
    process_webcam()
    
def select_images():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_paths:
        for file_path in file_paths:
            process_image(file_path)

# Update the GUI
def create_gui():
    root = Tk()
    root.title("YOLO Vehicle Detection")
    root.geometry("400x200")

    Label(root, text="YOLO Vehicle Detection", font=("Arial", 16)).pack(pady=10)

    Button(root, text="Select Images", command=select_images, width=20).pack(pady=5)
    Button(root, text="Select Video", command=select_video, width=20).pack(pady=5)
    Button(root, text="Start Webcam", command=start_webcam, width=20).pack(pady=5)

    Button(root, text="Exit", command=root.quit, width=20).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()