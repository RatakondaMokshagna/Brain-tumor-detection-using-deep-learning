from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
from transformers import pipeline  # Hugging Face text generation

# Initialize Flask app
app = Flask(__name__)

# Load trained CNN model for tumor classification
class BrainTumorCNN(torch.nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(64 * 56 * 56, 128)
        self.fc2 = torch.nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
model = BrainTumorCNN()
model.load_state_dict(torch.load('custom_brain_tumor_model.pth', map_location=torch.device('cpu')))
model.eval()

# Class labels
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def generate_text_llm(tumor_type):
    """Generate a short and clear medical explanation using Hugging Face's model."""
    
    # Updated prompts for better response
    medical_explanations = {
        "Glioma": "Gliomas are tumors that arise in the glial cells of the brain or spinal cord. They can be low-grade (slow-growing) or high-grade (aggressive). Treatment depends on the type and stage.",
        "Meningioma": "Meningiomas are typically benign tumors that develop in the meninges, the protective layers of the brain and spinal cord. They grow slowly and may not always require treatment unless they cause symptoms.",
        "No Tumor": "No signs of a tumor were detected in this MRI scan. If symptoms persist, further clinical evaluation may be needed.",
        "Pituitary": "Pituitary tumors occur in the pituitary gland, affecting hormone production. Most are benign but can cause hormonal imbalances requiring medical or surgical treatment."
    }

    # Use predefined explanations or fallback to a generated response
    if tumor_type in medical_explanations:
        return medical_explanations[tumor_type]
    
    # If no predefined text, generate one dynamically
    prompt = f"What is {tumor_type}? Provide a brief, medically accurate explanation."
    response = text_generator(prompt, max_length=80, do_sample=True)
    return response[0]["generated_text"]

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process the uploaded image and return a prediction with an LLM-generated explanation."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Process image
    img = Image.open(file).convert('RGB')
    img = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        result = class_names[predicted.item()]

    # Generate explanation using Hugging Face model
    generated_text = generate_text_llm(result)

    return jsonify({'prediction': result, 'generated_text': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
