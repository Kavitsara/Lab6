from flask import Flask, request, render_template, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io
import base64
import os
import json
from datetime import datetime

app = Flask(__name__)

# Load the model
model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = torch.nn.Linear(1280, 3)  # Changed to 3 classes
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    return preprocess(img).unsqueeze(0)

# File to store history
HISTORY_FILE = 'prediction_history.json'

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

@app.route('/', methods=['GET', 'POST'])
def index():
    history = load_history()

    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        img_bytes = file.read()
        img_tensor = preprocess_image(img_bytes)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = outputs.max(1)

        labels = ["Fungi/Bacteria", "Healthy", "Nutrient"]
        predicted_label = labels[predicted.item()]

        # Encode the image to base64 for displaying in HTML
        encoded_img = base64.b64encode(img_bytes).decode('utf-8')
        img_data = f"data:image/jpeg;base64,{encoded_img}"

        # Add new data to history
        history.append({
            'filename': file.filename,
            'img_data': img_data,
            'label': predicted_label,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        save_history(history)

    return render_template('index.html', history=history)

if __name__ == '__main__':
    app.run(debug=True)