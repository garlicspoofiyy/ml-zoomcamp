from flask import Flask, render_template, request, redirect, url_for
import torch
from torchvision import transforms, models
from PIL import Image
import os
import torch.nn as nn

app = Flask(__name__, template_folder='templates')

def create_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model

# Get the absolute path to the model file using the app's root path
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
model_path = os.path.join(model_dir, 'pneumonia_model.pth')

# Verify the model file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the model with explicit map_location
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB') 
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return "Pneumonia" if predicted.item() == 1 else "Normal"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            upload_folder = 'uploads'
            os.makedirs(upload_folder, exist_ok=True)
            
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            result = predict_image(file_path)
            os.remove(file_path)
            return render_template('index.html', result=result)
    return render_template('index.html', result='')

if __name__ == '__main__':
    app.run(debug=True)