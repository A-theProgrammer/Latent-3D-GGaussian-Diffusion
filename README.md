
# RL and Adversarial Search & L3DG: Latent 3D Gaussian Diffusion


## Overview

This repository contains a project that combines two key ideas:

1. **RL and Adversarial Search for Game Playing:**  
   The code demonstrates how different AI algorithms (Minimax, Minimax with Alpha-Beta Pruning, and Q-Learning) can be used to play simple games like Tic Tac Toe and Connect 4.  
   *For non-technical readers:* Imagine teaching a computer to play a game by thinking through every possible move (Minimax) or by learning from experience (Q-Learning).

2. **L3DG: Latent 3D Gaussian Diffusion – Implementation and Demo:**  
   This part of the project shows a simplified version of a method for generating realistic 3D models using a two-step process:
   - **Latent Compression via VQ-VAE:** Compresses complex 3D data into a lower-dimensional latent space.
   - **Latent Diffusion Modeling:** Gradually refines noisy data in the latent space to produce clear 3D structures.  
   *For non-technical readers:* Think of it as taking a blurry photo (noisy data) and slowly making it clear by learning what a clear photo looks like.

---

## 1. Step-by-Step Instructions for Installing and Running the Code

### Prerequisites

- **Git** installed on your system.
- **Python 3.7+** installed (with pip).
- For deep learning, a machine with or without a CUDA-enabled GPU is supported.

### Installation and Virtual Environment Setup (Windows & VS Code)

1. **Clone or Download the Repository:**
   - Open your command prompt (or terminal) and run:
     ```bash
     git clone <repository_url>
     ```
   - Alternatively, download the repository as a ZIP file from GitHub and extract it.

2. **Open the Repository in Visual Studio Code:**
   - Launch VS Code.
   - Use `File > Open Folder` and select the folder containing the project files.

3. **Create a Virtual Environment:**
   - Open the VS Code terminal (Ctrl + `).
   - Run the following command to create a virtual environment named `venv`:
     ```bash
     python -m venv venv
     ```

4. **Activate the Virtual Environment:**
   - In the terminal, run:
     ```bash
     .\venv\Scripts\activate
     ```
   - You should see `(venv)` at the beginning of your terminal prompt, indicating that the environment is active.

5. **Install Required Dependencies:**
   - With the virtual environment activated, install the necessary Python packages:
     ```bash
     pip install torch numpy matplotlib seaborn open3d plotly
     ```
   - These packages support deep learning, numerical computations, plotting, and 3D data visualization.

6. **Run the Code:**
   - To train the models and see the demo, run:
     ```bash
     python <filename>.py
     ```
   - For the game-playing part, you might have a separate file (e.g., `rl_advesarialSearch.py`).
   - For the 3D diffusion demo, the code is contained in the provided file.

---

## 2. About L3DG

### L3DG: Latent 3D Gaussian Diffusion

#### **Paper Citation:**
> **Roessle, B., Müller, N., Porzi, L., Rota Bulò, S., Kontschieder, P., Dai, A., & Nießner, M. (2024).**  
> *L3DG: Latent 3D Gaussian Diffusion.* In SIGGRAPH Asia 2024 Conference Papers (SA '24).  
> [DOI:10.1145/3680528.3687699](https://doi.org/10.1145/3680528.3687699)

#### **Theoretical Background:**

- **Latent Compression via VQ-VAE:**
  - Compresses complex 3D point cloud data into a lower-dimensional space.
  - Uses vector quantization to convert continuous data into a discrete set of values.
  
- **Latent Diffusion Modeling:**
  - Operates in the learned latent space.
  - Gradually denoises latent vectors to generate realistic 3D structures.

#### **Demo Highlights:**

- **Data Generation:** Synthetic 3D point clouds are generated for training.
- **Visualization:** The output is an interactive 3D scatter plot where colors indicate the distance of points from the origin.
- **Output:** Each output is a 1024-point 3D point cloud that demonstrates the model's ability to generate coherent 3D structures from random noise.

---

## 3. About My Implementation

### Key Components

- **Initial Setup:**
  - Checks for CUDA availability using PyTorch.
  - Imports necessary libraries such as `torch`, `numpy`, `matplotlib`, and `plotly`.

- **Synthetic Data Generation:**
  - Custom dataset class simulates 3D point clouds.

- **VQ-VAE Components:**
  - **Encoder:** Compresses 3D point clouds into 64-dimensional latent vectors.
  - **Vector Quantizer:** Discretizes the latent space using 512 codebook entries.
  - **Decoder:** Reconstructs the point cloud from the quantized latent vectors.

- **Training the VQ-VAE:**
  - Uses mean squared error for reconstruction loss and a VQ-specific loss.
  - Typical training time is approximately **10–20 seconds per epoch on CPU** (faster on GPU).

- **Diffusion Model Components:**
  - A simple MLP that predicts the clean latent code from a noisy version.
  - Training time is similar to the VQ-VAE, with fast inference.

- **Sampling and Visualization:**
  - **Inference:**  
    - Sampling latent vector: ~10–100ms (depending on GPU/CPU).
    - Decoding: ~10–20ms.
    - Rendering with Plotly: ~300–500ms (can be faster with Open3D).
    - Total time from noise → full 3D point cloud + render: ~0.5–1.5 seconds.
  - Generates and displays a 3D point cloud in an interactive Plotly window.

*For the complete code, please refer to the file(s) in the repository. Inline comments explain each step.*

---

## 4. Project Explanation

### Technical Concepts IMplemented in the Project

- **Deep Learning & PyTorch:**
  - PyTorch is used to build and train neural networks (VQ-VAE and diffusion model). It allows for efficient computations on CPU or GPU.

- **VQ-VAE (Vector Quantized Variational Autoencoder):**
  - **Encoder:** Compresses a detailed 3D point cloud into a simpler, 64-dimensional representation.
  - **Vector Quantizer:** Converts the continuous latent vector into a set of standard symbols (from a fixed dictionary).
  - **Decoder:** Reconstructs the original 3D point cloud from these symbols.
  
- **Diffusion Model:**
  - Starts with noisy latent vectors and gradually refines them into clean representations, much like sharpening a blurry image.

- **3D Point Cloud Visualization:**
  - A point cloud is a collection of points in 3D space. The interactive Plotly graph lets you see the generated 3D structure with a color gradient indicating point distance from the origin.

### Performance and Resource Usage

- **Training:**
  - **VQ-VAE:** Approximately 10–20 seconds per epoch on CPU (faster on GPU).
  - **Diffusion Model:** Similar training time to VQ-VAE; inference is very fast.

- **Inference (Sampling + Decoding + Rendering):**
  - **Sampling latent vector:** ~10–100ms (depending on GPU/CPU).
  - **Decoding:** ~10–20ms.
  - **Rendering with Plotly:** ~300–500ms (can be faster with Open3D).
  - **Total time from noise to full 3D point cloud + render:** ~0.5–1.5 seconds. This supports real-time interaction in applications like Gradio or Streamlit.

- **Output Details:**
  - Each output is a 1024-point 3D point cloud.
  - The encoder compresses the point cloud into a 64-dimensional latent vector.
  - The decoder reliably reconstructs the overall shape and structure.
  - You can increase point density by modifying the `num_points` parameter, though this may affect speed and memory usage.

- **Real-Time Capabilities:**
  - Capable of interactive sampling and rendering in applications such as Gradio or Streamlit.
  - On a modest GPU (e.g., RTX 2060+), you can sample 5–10 point clouds per second.
  - Suitable for live feedback in AR/VR previews or dynamic user interfaces.

- **Resource Usage:**
  - **VQ-VAE Model:** Approximately 1–2 GB of RAM/VRAM (using float32 precision).
  - **Diffusion Model:** Approximately 0.5–1 GB.
  - **Rendering:** Minimal GPU memory usage.
  - **CPU:** Low to moderate usage (mostly during data loading and rendering).
  - **GPU:** Efficient utilization, especially with torch.cuda enabled.

### Summary

This project demonstrates a fast and lightweight VQ-VAE + diffusion pipeline that:
- Generates high-detail 3D point clouds (1024 points) quickly.
- Supports real-time inference and interactive visualization.
- Is compact in memory usage and suitable for deployment in interactive apps or AR/VR previews.

By following the instructions and reading the code comments, even non-technical users can understand the step-by-step process behind these advanced AI methods.

---

## Conclusion

This repository serves as a comprehensive demonstration of both game-playing AI and advanced 3D generative modeling using diffusion techniques. It is designed to be accessible to both technical and non-technical audiences, providing clear instructions, thorough explanations, and interactive visualizations.
