# Track-fix: Explainable Railway Surface Defect Classification

An industrial-grade **explainable deep learning system** for automated railway surface defect classification using Gradient Class Activation Mapping.  
The system classifies railway track images into **Defective** and **Non-Defective** categories and generates **Grad-CAM heatmaps** to explain model decisions.

---

## 📌 Project Overview

This project implements a convolutional neural network (CNN) based crack detection pipeline using **ResNet18** with transfer learning.  
Explainability is achieved using **Gradient-weighted Class Activation Mapping (Grad-CAM)**, enabling visualization of regions influencing the classification output.

---

## 🎯 Objectives

- Develop an accurate railway crack classifier  
- Apply transfer learning using pretrained CNNs  
- Integrate explainable AI (Grad-CAM)  
- Support batch image processing  
- Provide industrial-style result logging  

---

## 🧠 Model Architecture

- Backbone: ResNet18 (pretrained on ImageNet)  
- Modified final layer: 2-class output  
- Input size: 224 × 224 RGB  

---

## 🧰 Frameworks & Libraries

- PyTorch  
- Torchvision  
- OpenCV  
- NumPy  
- Pillow  
- Scikit-learn  
- pytorch-grad-cam  

---

## 📁 Dataset Structure

Dataset/
├── Train/
│ ├── Defective/
│ └── Non_defective/
├── Validation/
│ ├── Defective/
│ └── Non_defective/
└── Test/


Binary labels:

- 0 → Defective  
- 1 → Non_defective  

Dataset size: ~2 GB

---

## 🔁 Workflow Pipeline

Dataset
↓
Preprocessing
↓
ResNet18 Training
↓
Model Saved (.pth)
↓
Image Testing
↓
Grad-CAM Heatmap Generation
↓
Batch Processing & Reports


---

## 🏋️ Training Configuration

- Epochs: 10  
- Batch size: 16  
- Optimizer: Adam  
- Learning rate: 0.0001  
- Loss function: Cross-Entropy  
- Device: CPU  

---

## 📉 Training Log

| Epoch | Loss |
|------:|------|
| 1 | 0.4892 |
| 2 | 0.0797 |
| 3 | 0.0343 |
| 4 | 0.0152 |
| 5 | 0.0049 |
| 6 | 0.0057 |
| 7 | 0.0038 |
| 8 | 0.0097 |
| 9 | 0.0032 |
| 10 | 0.0030 |

<img width="1223" height="715" alt="image" src="https://github.com/user-attachments/assets/ee5c74bf-86c4-4272-816c-838c5a305f57" />


---

## ✅ Validation Performance

Validation Accuracy: **93.55%**

---

## 🧪 Testing Procedure

1. Place image(s) inside `test/` folder  
2. Run prediction script  
3. Observe predicted class  
4. Run Grad-CAM script  
5. Inspect generated heatmaps  

---

## 🔍 Explainability (Grad-CAM)

- Heatmaps highlight **regions influencing model decisions**
- Heatmap presence does **not necessarily indicate a defect**
- For non-defective images, model highlights structural regions (rails, fasteners) to confirm normal condition

---

## 🧪 Experimental Results

### Defective Images

- 150 images tested  
- 150 correctly classified  
- Accuracy: 100%

  ![images (3)_heatmap](https://github.com/user-attachments/assets/18d1b414-fddd-400d-9756-4f86713a7fd0)
  ![images (1)_heatmap](https://github.com/user-attachments/assets/b18e0f61-3316-4e91-b271-5c67dd3d177b)
![89e8b1a9ca8c41919021f8f4faaf37f0_heatmap](https://github.com/user-attachments/assets/19ee38f6-f87c-4bd6-9629-f261c3b109b3)
![2007_heatmap](https://github.com/user-attachments/assets/e1850d69-d268-49af-8883-7c7c6f9e9c59)



### Non-Defective Images

- 150 images tested  
- 150 correctly classified  
- Accuracy: 100%  

Heatmaps appeared on both classes, representing attention regions.

![IMG_20201114_100139_heatmap](https://github.com/user-attachments/assets/16ebcfd3-d5ad-4a67-84f3-b1566ee54eb8)
![IMG_20201114_101019_heatmap](https://github.com/user-attachments/assets/d5fc494e-0f28-4841-9fb6-073f273d04e4)
![IMG_20201114_100310_heatmap](https://github.com/user-attachments/assets/3d45c898-d8ec-4f19-97a6-a94611fa4f03)
![IMG_20201114_101046_heatmap](https://github.com/user-attachments/assets/51d0ebc1-36ff-471c-9296-179986057b61)

![alt text](image.png)


---

## 🏭 Industrial Comparison

| System Type | Typical Accuracy |
|------------|----------------|
| Traditional CV | 60–75% |
| Basic CNN | 80–90% |
| Industrial Prototype | 90–95% |
| Safety-Certified Systems | 95–99% |

This project: **93.55% (Industrial Prototype Grade)**

---

## 📦 Model Artifact

- File: `rail_crack_model.pth`  
- Size: ~45 MB  

---

## ⚠️ Limitations

- Classification only (no bounding boxes)  
- Grad-CAM attention may be broad  
- CPU training time high  

---

## 🚀 Future Work

- YOLO-based crack localization  
- Video stream processing  
- Hyperparameter tuning  
- GUI dashboard  
- Edge deployment  

---

