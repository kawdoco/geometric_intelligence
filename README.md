# Geometric Intelligence
# Geometry-Lensing & U-Net (CPU)

Inline comparison, input gallery, and pipeline:

<p align="center">
  <img src="comparison.png" alt="Comparison" width="48%">
  <img src="input_gallery(1).png" alt="Input Gallery" width="48%"><br/>
  <img src="methodology_pipeline_clean_h(1).png" alt="Methodology Pipeline" width="80%">
</p>

## Contents
- `unet_cpu_train.py` — Train a U-Net on CPU  
- `unet_cpu_infer.py` — Run CPU inference and save masks/overlays  
- `lensing_without_ai.py` — Simple gravitational-lensing style demo (no AI)  
- `comparison.png`, `input_gallery(1).png`, `methodology_pipeline_clean_h(1).png` — figures used in this README

---

## 1) Step 1

```bash
# 1) Clone
git clone kawdoco/geometric_intelligence
cd geometry-lensing-unet

# 2) (Optional) create a venv
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

## 2) Step 2 Install (CPU-only)
# (venv) python -m pip install --upgrade pip
# (venv) python -m pip install numpy matplotlib pillow

# Then run with the venv’s python:

# (venv) python lensing_without_ai.py

# Check the auto created folder and check images that you generated

## 3) Step 3 – play with CPU AI Install Packages

# (venv) python -m pip install --upgrade pip
# (venv) python -m pip uninstall -y torch numpy
# (venv) python -m pip install "numpy==1.26.4”
# CPU-only torch wheel (works on Mac/CPU)
# (venv) python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then run with the venv’s python:Train on CPU (small & fast)

# (venv) python unet_cpu_train.py --epochs 5 --samples 512 --size 128 --noise 0.02

## 4) Step 4 - inference with CPU AIInference demo (CPU)

# python unet_cpu_infer.py --size 128
# Now check newly created folders and images , please kindly fix your python errors by yourself.
# This will create an outputs_unet/ folder
# with:tiny_unet_cpu.pth (weights)train_curve.png (loss vs epoch)sample_train_pred.png
# (target κ vs predicted κ̂ from training set)infer_pred.png (γ₁, γ₂, and predicted κ̂ for a new sample)

<br>
<img width="895" height="326" alt="image" src="https://github.com/user-attachments/assets/c21d7d59-8908-4546-af60-349f42c90d59" />

<br>
<img width="991" height="296" alt="image" src="https://github.com/user-attachments/assets/51d07388-00af-42b6-9b98-0c76e2779c25" />


<br>
<img width="464" height="296" alt="image" src="https://github.com/user-attachments/assets/32fa6275-181d-4508-85f4-72c353abd6c7" />



python unet_cpu_infer.py -h
python lensing_without_ai.py -h
