# AGENTS.md

## Cursor Cloud specific instructions

### Project overview
LKFormer is a deep learning research project for infrared image super-resolution using PyTorch. The main model architecture is in `SRModel/LKFormer.py` (class `Restormer`). Training entry point is `main_train_SR.py`; testing entry point is `main_test.py`.

### Key caveats
- **No GPU in Cloud VM**: The codebase is designed for CUDA GPU training/inference. The cloud VM has no GPU, so only CPU-based forward passes and model verification are possible. For GPU-dependent code (training, full testing), use `torch.device('cpu')` instead of `torch.device('cuda:0')`.
- **Incomplete codebase**: Several local modules referenced by the codebase are missing from the repository: `data/`, `utilss/`, `swinUnetSR/`, `blindSR/`. These are from the KAIR framework (https://github.com/cszn/KAIR). Only `utils/module_util.py` has been stubbed out to enable core model imports.
- **timm version pinned**: Must use `timm==0.6.13` — newer versions break imports like `timm.models.layers.helpers.to_2tuple` and `timm.models.registry.register_model`.
- **Main importable models**: `SRModel/LKFormer.py` (Restormer) and `models/network_swinir.py` (SwinIR) are fully importable and can run forward passes on CPU. Other models (e.g. `network_hybrid.py`) have cross-dependency issues from missing modules.
- **No automated test suite**: The repository has no test framework or test files. Validation is done via training/testing scripts which require datasets and GPU.

### Running the models (CPU smoke test)
```python
import torch
from SRModel.LKFormer import Restormer

model = Restormer(inp_channels=3, out_channels=3, dim=48, upscale=2)
model.eval()
x = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    out = model(x)
# out.shape == (1, 3, 64, 64)
```

### Linting
```bash
flake8 --max-line-length=200 --select=E9,F63,F7,F82 .
```

### Dependencies
See `requirements.txt`. Install with `pip install -r requirements.txt`. For CPU-only environments, install PyTorch CPU build first: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
