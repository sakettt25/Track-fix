import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("rail_crack_model.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# -----------------------------
# LOAD IMAGE
# -----------------------------
IMAGE_PATH = "test_image.png"
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# -----------------------------
# PREDICTION
# -----------------------------
with torch.no_grad():
    outputs = model(input_tensor)
    prediction = torch.argmax(outputs, 1).item()

if prediction == 0:
    print("Prediction: DEFECTIVE")
else:
    print("Prediction: NON-DEFECTIVE")

# -----------------------------
# GRAD-CAM SETUP
# -----------------------------
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

grayscale_cam = cam(input_tensor=input_tensor)[0]

# -----------------------------
# HEATMAP OVERLAY
# -----------------------------
rgb_image = np.array(image.resize((224, 224))) / 255.0
heatmap = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
cv2.imwrite(
    "gradcam_output.jpg",
    cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
)

print("Heatmap saved as gradcam_output.jpg")
