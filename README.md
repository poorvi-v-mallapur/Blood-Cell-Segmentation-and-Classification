# 🩸 Automated Blood Cell Detection Using Deep Learning (YOLO)

Automating the detection and counting of **Red Blood Cells (RBCs)**, **White Blood Cells (WBCs)**, and **Platelets** using the **YOLOv11n** deep learning model.  
This project aims to speed up hematology workflows and reduce human error in manual cell counting.

---

## 🚀 Features
- Detects RBCs, WBCs, and Platelets from microscopic images.
- Uses YOLOv11n for high-speed and high-accuracy object detection.
- Trained on the BCCD dataset with transfer learning.
- Visualizes annotated images with bounding boxes and confidence scores.
- Metrics: Precision, Recall, mAP@0.5.

---

## 🧠 Tech Stack
| Component | Technology Used |
|------------|----------------|
| Language | Python 3.8 |
| Deep Learning Framework | PyTorch |
| Object Detection | YOLOv11n (Ultralytics) |
| Image Processing | OpenCV |
| Training Environment | Google Colab (GPU) |
| Annotation Tool | LabelImg |
| Dataset | BCCD Dataset (Kaggle / Roboflow) |

---

Automated-Blood-Cell-Detection/
│
├── data/ # Dataset (train/test images)
├── notebooks/ # Colab notebooks for training/evaluation
├── models/ # Saved model weights (best.pt)
├── results/ # Output images with detections
├── requirements.txt # Python dependencies
├── detect.py # Detection/inference script
├── train.py # Model training script
└── README.md # You're here :)

---

## ⚙️ Setup & Installation

# 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Automated-Blood-Cell-Detection.git
cd Automated-Blood-Cell-Detection
```

# 2. Create a virtual environment (optional but recommended)
```
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```

# 3. Install dependencies
```
pip install -r requirements.txt
```

# 🧬 Training the Model

To train the YOLOv11n model on the BCCD dataset:
```
yolo train model=yolov11n.pt data=data.yaml epochs=50 imgsz=640
```
💡 Tip: Adjust epochs and batch size depending on GPU capacity.

# 🔍 Running Detection

To run inference on test images:
```
yolo detect predict model=runs/train/exp/weights/best.pt source=./test_images/
```
Detected images will be saved in:
```
runs/detect/exp/
```

---

## 📈 Future Improvements

- Classify WBC subtypes (lymphocytes, monocytes, etc.)
- Integrate disease prediction (malaria, leukemia, etc.)
- Deploy as a web or mobile app for labs.

---

## 🧾 References

- BCCD Dataset – Kaggle
- Ultralytics YOLOv11n
- PyTorch
- OpenCV
- Roboflow
