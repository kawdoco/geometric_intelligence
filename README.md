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

## 1) Quick Start

```bash
# 1) Clone
git clone kawdoco/geometric_intelligence
cd geometry-lensing-unet

# 2) (Optional) create a venv
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3) Install (CPU-only)
python -m pip install --upgrade pip
# Minimal set:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy opencv-python pillow matplotlib scikit-image tqdm pyyaml

# 4) Check help
python unet_cpu_train.py -h
python unet_cpu_infer.py -h
python lensing_without_ai.py -h
