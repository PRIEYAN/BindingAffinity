# GPU Setup Guide

## Quick Setup

1. **Install PyTorch with CUDA support** (if you want neural network GPU support):
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Check your CUDA version
nvidia-smi
```

2. **Install other requirements**:
```bash
pip install -r requirements.txt
```

3. **Verify GPU is detected**:
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## GPU Support by Model

### XGBoost
- **GPU Support**: Yes (automatic detection)
- **Method**: Uses `device='cuda'` or `tree_method='gpu_hist'`
- **Fallback**: Automatically falls back to CPU if GPU unavailable

### LightGBM
- **GPU Support**: Yes (automatic detection)
- **Method**: Uses `device='gpu'`
- **Fallback**: Automatically falls back to CPU if GPU unavailable

### PyTorch Neural Network
- **GPU Support**: Yes (automatic detection)
- **Method**: Uses `torch.device('cuda')`
- **Fallback**: Automatically falls back to CPU if GPU unavailable
- **Note**: Requires PyTorch with CUDA support installed

### Random Forest / Gradient Boosting (scikit-learn)
- **GPU Support**: No (CPU only)
- **Note**: These models don't support GPU acceleration

## Running with GPU

Simply run the script - it will automatically detect and use GPU if available:

```bash
python3 train_binding_affinity.py
```

The script will print which device is being used:
- `Using GPU for XGBoost`
- `Using GPU for LightGBM`
- `Using device: cuda` (for PyTorch)

## Troubleshooting

### XGBoost GPU not working
1. Make sure you have CUDA installed: `nvidia-smi`
2. XGBoost should automatically use GPU if CUDA is available
3. If it falls back to CPU, check XGBoost version: `pip install --upgrade xgboost`

### LightGBM GPU not working
1. LightGBM requires CUDA toolkit to be installed
2. Install CUDA toolkit if not already installed
3. LightGBM will automatically detect GPU if available

### PyTorch GPU not working
1. Check CUDA is available: `python3 -c "import torch; print(torch.cuda.is_available())"`
2. If False, reinstall PyTorch with CUDA support (see Quick Setup above)
3. Verify CUDA version matches: `nvidia-smi` shows CUDA version

## Performance Tips

1. **XGBoost and LightGBM**: GPU acceleration is most beneficial for large datasets (10k+ samples)
2. **PyTorch Neural Network**: GPU is essential for training neural networks efficiently
3. **Batch Size**: For PyTorch, adjust batch size in the script if you run out of GPU memory
