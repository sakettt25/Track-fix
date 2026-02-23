import os
import csv
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# -----------------------------
# SETTINGS
# -----------------------------
TEST_FOLDER = "test"
OUTPUT_FOLDER = "outputs/heatmaps"
CSV_FILE = "outputs/results.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("rail_crack_model.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],
                         [0.5,0.5,0.5])
])

# -----------------------------
# GRADCAM
# -----------------------------
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

# -----------------------------
# OPEN CSV
# -----------------------------
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Prediction", "Heatmap_File"])

    # -------------------------
    # PROCESS EACH IMAGE
    # -------------------------
    for file in os.listdir(TEST_FOLDER):

        if file.lower().endswith((".jpg",".png",".jpeg",".webp")):

            path = os.path.join(TEST_FOLDER, file)

            image = Image.open(path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            # Prediction
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output,1).item()

            label = "DEFECTIVE" if pred==0 else "NON-DEFECTIVE"

            # GradCAM
            grayscale_cam = cam(input_tensor=input_tensor)[0]
            rgb_img = np.array(image.resize((224,224))) / 255.0
            heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            out_name = file.split(".")[0] + "_heatmap.jpg"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)

            cv2.imwrite(out_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

            writer.writerow([file, label, out_path])

            print(f"{file} --> {label}")

print("\nBatch processing complete.")
print("Heatmaps saved in outputs/heatmaps/")
print("Results saved in outputs/results.csv")
