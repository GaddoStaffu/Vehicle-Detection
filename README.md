# YOLOv8 Car and Motorcycle Detection

This project utilizes **YOLOv8** to detect cars and motorcycles using the **N_O_2 Computer Vision Project** dataset from [Roboflow](https://universe.roboflow.com/nada-majd-e4uhb/n_o_2). The model is trained to recognize and differentiate between vehicles, making it useful for traffic monitoring and surveillance.

## 🚀 Features

- **Real-time detection** of cars and motorcycles using YOLOv8
- **Pre-trained and fine-tuned model** on the N_O_2 dataset
- **Supports video and image inference**
- **Easy deployment** with Python and OpenCV

## 📂 Dataset

The project is trained on the **N_O_2 Computer Vision Project** dataset, which consists of labeled images of cars and motorcycles in various conditions. The dataset can be accessed from [Roboflow](https://universe.roboflow.com/nada-majd-e4uhb/n_o_2).

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/yolo-car-or-motor-detection.git
   cd yolo-car-or-motor-detection
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv/Scripts/activate
   pip install -r requirements.txt
   ```

3. Test the model:
   ```bash
   python testmodel.py --image <image_path>
   ```

## 🏎️ Vehicle Detection

This project detects vehicles (cars and motorcycles) in images and videos using a fine-tuned YOLOv8 model. It is designed for real-time applications such as traffic monitoring and surveillance.
