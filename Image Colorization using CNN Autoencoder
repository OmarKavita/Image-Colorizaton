# Image Colorization using Convolutional Autoencoder (PyTorch)

## 📌 Overview
This project implements a **deep learning–based image colorization system** that converts **grayscale images into RGB color images** using a **convolutional autoencoder** built with **PyTorch**.

The model learns a **pixel-wise mapping** from grayscale inputs to color outputs using an **encoder–decoder CNN architecture with skip connections**, enabling effective reconstruction of spatial structure and color information.  
The project demonstrates an **end-to-end computer vision workflow**, including data loading, model training, evaluation, and visualization.

---

## 🎯 Problem Statement
Grayscale images lack color information, which is essential for visual interpretation.  
This project aims to **automatically infer realistic colors** from grayscale landscape images using a **data-driven deep learning approach**, without relying on handcrafted rules.

---

## 📂 Project Structure
image-colourization-autoencoder/
├── colouring-image/
│   ├── gray/        # Grayscale images
│   └── color/       # Corresponding RGB images
├── Untitled.ipynb   # Jupyter notebook with full implementation
├── README.md

---

## 🚀 Key Features
- Custom LandscapeDataset class for easy loading of paired grayscale and color images
- Data pipeline using PyTorch DataLoader and torchvision transforms
- Deep convolutional autoencoder model (encoder+decoder)
- Hardware acceleration: CUDA support if available
- Visualization routines for both training data and model predictions
- Progress monitoring with tqdm

---

## 🧠 Model Architecture
- **Input:** Grayscale image (1 × 150 × 150)
- **Encoder:**  
  - Convolutional layers with stride-based downsampling  
  - Increasing channel depth to capture high-level features
- **Bottleneck:**  
  - Compact latent representation of image semantics
- **Decoder:**  
  - Transposed convolutions for upsampling  
  - Skip connections to preserve spatial details
- **Output:** RGB color image (3 × 150 × 150)
- **Activations:**  
  - ReLU (hidden layers)  
  - Sigmoid (output layer)

---

## 📊 Training Configuration

| Parameter        | Value |
|------------------|-------|
| Image Size       | 150 × 150 |
| Batch Size       | 32 |
| Epochs           | 15 |
| Optimizer        | Adam |
| Learning Rate    | 0.001 |
| Loss Function    | Mean Squared Error (MSE) |
| Device           | CPU / CUDA |

---

## 🔍 Evaluation Strategy
- Model performance is evaluated on **unseen test data**
- **Mean Squared Error (MSE)** is used as the quantitative metric

---

## 🛠️ Tech Stack
- Python  
- PyTorch  
- Torchvision  
- NumPy  
- Matplotlib  
- tqdm  

---

## 👤 Author
**Kavita Omar**  
📧 Email: **kavitaomariitk24@gmail.com**
