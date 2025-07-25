# DeepSeis-Residual-Attention-U-Net-for-Seismic-Velocity-Inversion
Deep learning approach for seismic inversion using Attention U-Net and Residual Attention U-Net. Optimized with early stopping, weight decay, and learning rate tuning, achieving a test MAPE of 0.0457 in the Speed and Structure Challenge on ThinkOnward.
# Seismic Velocity Inversion with Residual Attention U-Net

This repository contains the implementation of a Residual Attention U-Net architecture for seismic velocity inversion, developed as part of the **Speed and Structure Challenge** on the ThinkOnward platform.

## Project Overview
The project focuses on predicting subsurface velocity models from seismic receiver data using advanced deep learning architectures. We experimented with various approaches including Generative Diffusion Models (GDM), Attention U-Net, and Residual Attention U-Net. The final model achieved a **test MAPE of 0.045**, significantly improving over baseline methods.

## Approaches Tried
- **Generative Diffusion Models (GDM):**
  - Initially used fixed beta (poor generalization).
  - Implemented cosine decay beta schedule.
  - Computationally expensive with no significant improvement, hence abandoned.

- **Attention U-Net:**
  - Learning rate: `1e-4`, no weight decay, max epochs: `100` with early stopping.
  - Achieved test MAPE: **0.058**.
  - With additional data: test MAPE **0.051**, validation MAPE **0.062**.

- **Residual Attention U-Net:**
  - Learning rate: `1e-4`, weight decay: `1e-5`, max epochs: `100`.
  - Achieved validation MAPE **0.0454**, test MAPE **0.045**.

## Repository Structure
```
.
├── data/                     # Data files (not included in repo)
├── notebooks/                # Jupyter notebooks for experiments
│   └── Unet_seismic.ipynb
├── src/                      # Source code
│   ├── dataset.py            # Dataset loader
│   ├── model.py              # Residual Attention U-Net implementation
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── utils.py              # Helper functions
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Results
- **Best Test MAPE:** `0.045` (Residual Attention U-Net).
- Abandoned FWI (Full Waveform Inversion) as it required significantly high computational resources.

## How to Run
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd seismic-velocity-inversion
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run training:
   ```bash
   python src/train.py
   ```
4. Evaluate the model:
   ```bash
   python src/evaluate.py
   ```

## Requirements
See `requirements.txt` for all dependencies.

## License
This project is for educational and research purposes only.
