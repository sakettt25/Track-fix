import torch
from torchvision import models

# Create a pretrained model (ImageNet weights) for better accuracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.to(device)

# Save the model
torch.save(model.state_dict(), "rail_crack_model.pth")
print("Pretrained model created and saved as rail_crack_model.pth")
print("This model uses ImageNet weights for improved accuracy!")
