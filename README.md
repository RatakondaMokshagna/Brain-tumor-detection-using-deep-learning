🧠 SEAIPROJECT – Brain Tumor Classification App (Flask + Docker + Kubernetes)
This project is a deep learning web application built with Flask to classify brain tumors using a custom PyTorch model. The application is containerized with Docker and deployed on a Kubernetes cluster using Minikube.

🚀 Features
🧠 Brain tumor image classification using deep learning

🔬 Custom CNN model trained on MRI datasets

⚙️ Flask backend with HTML UI for uploading MRI scans

🐳 Docker containerization

☸️ Kubernetes deployment using Minikube

⚙️ Getting Started
1. Prerequisites
Python 3.8+

Docker

Minikube (with Docker driver)

kubectl

2. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/seaiproject.git
cd seaiproject
3. Start Minikube
bash
Copy
Edit
minikube start --driver=docker
eval $(minikube docker-env)
4. Build and Deploy Using Makefile
bash
Copy
Edit
make        # or: make all
5. Access the App
bash
Copy
Edit
make url
Open the URL shown in your browser.

