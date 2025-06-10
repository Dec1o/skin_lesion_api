from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Definir modelo idêntico
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN()
model.load_state_dict(torch.load("lesion_model.pth", map_location=device))
model.to(device)
model.eval()

CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

def preprocess_image(image):
    image = image.convert("RGB").resize((28, 28))
    img_array = np.array(image).transpose(2,0,1) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)
    return img_tensor.to(device)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Envie uma imagem com o campo 'image'"}), 400
    file = request.files["image"]
    image = Image.open(file.stream)
    img_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        prob = torch.softmax(outputs, dim=1)[0][predicted].item()

    return jsonify({
        "class": CLASSES[predicted.item()],
        "confidence": prob
    })

if __name__ == "__main__":
    app.run(debug=True)
