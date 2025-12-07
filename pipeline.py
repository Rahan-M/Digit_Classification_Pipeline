import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

best_model_path="best_model.pth"

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def preprocess_image(image_path):
    # Load image with PIL
    img = Image.open(image_path).convert("L")  # convert to grayscale

    # Resize to MNIST size
    img = img.resize((28, 28))

    # Convert to numpy array (0–255)
    img = np.array(img).astype(np.float32)

    # Normalize to 0–1
    img = img / 255.0

    # Add channel and batch dims: (1, 1, 28, 28)
    img = np.expand_dims(img, axis=0)  # channel
    img = np.expand_dims(img, axis=0)  # batch

    return torch.tensor(img, dtype=torch.float32)

def predict_digit(image_path, model, device="cpu"):
    img_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        predicted = torch.argmax(outputs, dim=1).item()

    return predicted

def load_trained_model(model_path, device):
    model = MNIST_CNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=load_trained_model(best_model_path, device)
    image_path=input("Input Image Path : ")
    print(f"The digit in image is : {predict_digit(image_path, model, device)}")