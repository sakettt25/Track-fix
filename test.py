import torch
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load trained model
# -----------------------------
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("rail_crack_model.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Image transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# -----------------------------
# Load image
# -----------------------------
img_path = "test_image.png"
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output, 1).item()

if pred == 0:
    print("Prediction: DEFECTIVE")
else:
    print("Prediction: NON-DEFECTIVE")
