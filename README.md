# Brain Tumor Classification Web Application (Flask, Docker, Kubernetes)

## Overview

This project is a deep learning-based web application designed to classify brain tumors from MRI images. The system uses a custom Convolutional Neural Network (CNN) implemented in PyTorch and provides predictions through a web interface built with Flask.

The application is containerized using Docker and deployed on a Kubernetes cluster using Minikube. The project demonstrates how a machine learning model can be integrated into a web application and deployed using modern containerization and orchestration technologies.

## Features

- Brain tumor classification from MRI images using deep learning
- Custom CNN model implemented in PyTorch
- Web interface for uploading MRI images and viewing predictions
- Backend developed using Flask
- Docker-based containerization for reproducible environments
- Kubernetes deployment using Minikube for scalable application management

## Technologies Used

**Programming Language**
- Python

**Machine Learning Framework**
- PyTorch

**Web Framework**
- Flask

**Containerization and Deployment**
- Docker
- Kubernetes (Minikube)

**Libraries**
- NumPy
- OpenCV
- pandas
- scikit-learn
- matplotlib

## System Architecture

The system consists of the following components:

### Deep Learning Model
A custom convolutional neural network trained on MRI brain tumor datasets.  
The model performs classification of tumor types based on MRI images.

### Web Application
A Flask-based backend that handles image uploads and model inference.  
A simple HTML interface allows users to upload MRI scans and receive predictions.

### Containerization
The application is packaged using Docker to ensure consistent execution across environments.

### Orchestration
The Docker container is deployed on a Kubernetes cluster using Minikube for managing application deployment.

## Prerequisites

Before running the application, ensure the following tools are installed:

- Python 3.8 or higher
- Docker
- Minikube
- kubectl

## Installation

Clone the repository:

```bash
git clone https://github.com/RatakondaMokshagna/seaiproject.git
cd seaiproject
