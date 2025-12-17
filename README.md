# 🍀 Intreeligent: A Clustering Model of Tree Crowns for Enhanced Ecological Monitoring

*Juanico, Canaman, Mosqueda, Quiacao, Sanchez*

---
🍁 Intreeligent is a deep learning framework designed to automate the analysis of complex tropical forests using Unmanned Aerial Vehicle (UAV) RGB imagery. This project addresses the resource constraints of ecological monitoring for diverse forest by evaluating and implementing unsupervised clustering methods for Individual Tree Crown (ITC) structural assessment.

---
### 📖 Table of Contents
- [✴️ The Lore](#-the-lore)
- [👨‍🔬 Methodology](#-methodology)
- [✨ Key Features](#-key-features)
- [🤖 Tech Stack](#-tech-stack)
- [📊 Results](#-results)
- [🖥 Usage](#-usage)
- [👥 The Team](#-the-team)
- [Acknowledgments](#acknowledgments)

---
## ✴️ The Lore

Ecological monitoring is challenging so as acquiring labels in a data-constrained contexts such as the Philippines. While supervised learning models are accurate, this require extensive labeled datasets which is challenging to acquire. Intreeligent introduces a framework that utilizes unsupervised clustering of deep features to reveal meaningful structural patterns without requiring species-level labels.

---

##  👨‍🔬 Methodology

This project compares supervised and unsupervised processes across three stages where:
1. **Data Acqusition:** High-resolution imagery was acquired via UAV captures across Misamis Oriental and Bukidnon. External datasets are used for added training and evaluation.
2. **Object Delineation (Localization):**
	1. Evaluated: Unsupervised Superpixels (SLIC, LSC) vs. Supervised Instance Segmentation (Mask R-CNN)
	2. Selected: **Mask R-CNN** (ResNet50 FPN) was selected as the localizer due to superior performance in handling overlapping canopies.
3. **Feature Extraction (Autoencoders):**
	1. Latent space extraction was performed using three backbone architectures:
		1. ResNet34
		2. ResNet50
		3. DINOv2 (Vision Transformer-based Self-Supervised Learning)
4. **Clustering:**
	1. Comparison of **DBSCAN** vs. **HDBSCAN** where the latter was selected due to its ability to handle varying cluster densities inherent in tropical forests.


---

## ✨ Key Features

1.  **Automated Tree Localization:** Uses a fine-tuned Mask R-CNN to detect individual tree crowns in dense orthomosaic maps.
2.  **Density-Based Clustering:** Groups trees based on visual and structural similarity using HDBSCAN, adaptable to noise and varying densities.
3.  **Interactive Frontend:** A Streamlit-based web application for users to upload imagery, customize clustering parameters, and visualize results interactively via Plotly .

---

## 🤖 Tech Stack

- **Language**: Python
- **Deep Learning**: PyTorch, PyTorch Lightning, Torchvision
- **Computer Vision**: Detectron2 (Mask R-CNN), OpenCV, Pillow
- **Clustering**: Scikit-learn(DBSCAN), hdbscan, umap-learn
- **Visualization**: Plotly, Matplotlib
- **App Framework**: Streamlit
- **Experiment Tracking**: Weights and Biases (WandB)
- **Cloud Platform (for training)**: AWS Sagemaker (m4.xlarge instance)

---

## 📊 Results

To benchmark performance, over 18,000 tree crown masks are used where:

**Segmentation**
- **Mask R-CNN** performed significantly over unsupervised methods (SLIC/LSC) which failed to capture meaningful crown boundaries.
- Best performance achieved at **Epoch 5** with an **mAP@50 of 0.305**.

**Clustering and Feature Extraction**
- **DINOv2** achieved the highest external validation scores **(ARI: 0.42, NMI: 0.71)**, proving that self-supervised transformer features align best with ground truth structural boundaries.
- **ResNet 50**, while having lower metric scores, produced visually coherent clusters for species identity upon manual inspection.
- **HDBSCAN** performs better than DBSCAN, which effectively idenfity up to 225 distinct clusters per benchmarking and handling high noise levels of diverse forest data.

---

## 🖥 Usage

#### 🌐 Web App
To access **Intreeligent**, click the link provided below:

➡️ https://huggingface.co/spaces/fmmkii/intreeligent?logs=container

### 🖥 Local Use

**Prerequisites** (Minimum Requirements)
- Python 3.8+
- 8 GBs of RAM
- CUDA-enabled GPU (Most recommended if available)

**Setup**
1. Clone the repository
```bash
git clone https://huggingface.co/spaces/fmmkii/intreeligent
```

2. Install the dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
streamlit run app.py
```

#### 🔧 Own Training

**Setup**
1. Clone the repository
```bash
https://github.com/rocketwolf98/intreeligent-v1.git
```

Inside the directory, this contains multiple Jupyter Notebooks containing Clustering Pipeline, Inference Pipeline, Localizer Pipeline, and the Full Pipeline. Modifications are always welcome.

---

## 👥 The Team

This project was submitted to the **University of Science and Technology of Southern Philippines (USTP)** - Cagayan de Oro as a requirement for the degree of Bachelor of Science in Data Science (2025).

- **Hernel Niño C. Juanico**
- **Rachel Jasmine Canaman**
- **Visaviern V. Mosqueda**
- **Airyll H. Sanchez**
- **Aubrey Rose J. Quiacao**

---
### Acknowledgments

- **DENR Region-X** for providing the UAV orthomosaic imagery.
- **Roboflow Universe** for supplementary training datasets.
- **AWS** for computational resources via Sagemaker.
