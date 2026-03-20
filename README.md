# Conditional Face Generation with Diffusion Models (CelebA)

This repository contains the implementation of a **Conditional Diffusion Model** designed to generate realistic 64x64 face images using the **CelebA** dataset. Developed as part of a Generative AI course, the project focuses on advanced scheduling, conditioning strategies, and optimized U-Net architectures.

---

## 🚀 Project Overview
The core objective is to generate human faces that satisfy specific combinations of three binary attributes:
1. **Male / Female** (Attribute #20)
2. **Smiling / Not Smiling** (Attribute #31)
3. **Young / Not Young** (Attribute #39)

The model samples images starting from pure Gaussian noise and guides the denoising process toward the target class using **Classifier-Free Guidance (CFG)**.

## 🛠️ Neural Network Architecture
The architecture is based on a custom **U-Net** featuring several state-of-the-art modules:

* **Adaptive Group Normalization (AdaGN):** Dynamic injection of context (time + attributes) via learned scale ($\gamma$) and shift ($\beta$) parameters for each residual block.
* **Residual Blocks:** Implementation of internal skip connections to prevent gradient vanishing and allow the network to learn incremental denoising transforms.
* **Time Encoding:** Sinusoidal positional embeddings for the time step $t$, enabling the model to distinguish between different noise levels (from coarse structure to fine detail).
* **Conditioner MLP:** A dedicated module that fuses the *Time Embedding* (64-dim) with the *Attribute Labels* (3-dim) into a shared 256-dimensional latent space.

## 📈 Training and Scheduling Techniques
To optimize generative quality and training stability, the following solutions were implemented:

* **Cosine Noise Schedule:** A cosine-based noise trajectory to distribute image degradation more uniformly across the 1000 steps, proving more efficient than the standard linear schedule.
* **Dataset Balancing:** Used a `WeightedRandomSampler` to balance the 8 possible attribute combinations within the CelebA dataset, ensuring minority classes are adequately represented during optimization.
* **Data Augmentation:** A pre-processing pipeline including `CenterCrop` for face localization, `RandomHorizontalFlip` for spatial invariance, and normalization to the [-1, 1] range.

## 🧪 Inference and Results
The model supports sampling via **DDIM (Denoising Diffusion Implicit Models)**, which allows for high-quality generation with significantly fewer steps compared to the standard DDPM approach.

### Generating Samples
Within the `DifussionModel.ipynb` notebook, the `generate_and_save_grid` function produces a validation grid showing all 8 possible attribute combinations.

> **Technical Note:** During inference, a guidance scale (e.g., $w = 4.0$) is used to amplify the conditioning signal and improve attribute fidelity.

## 📋 Requirements
* Python 3.10+
* PyTorch & Torchvision
* NumPy
* Matplotlib

## 📂 Repository Structure
* `DifussionModel.ipynb`: Main notebook containing the architecture, training loop, and generation logic.
* `24 Project assignment.pdf`: Project specifications and academic requirements.
* `diffusionModels.pdf` / `guidedDiffusion.pdf`: Theoretical background on DDPM and Guided Diffusion.
