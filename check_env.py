"""
check_env.py — проверка Фазы 0: PyTorch, CUDA, имя видеокарты и VRAM + тест на GPU.
"""


"""Phase 0 check: confirm Python sees the RTX 3090 and its ~24 GB.

Run inside the `bioai` conda environment:
    python check_env.py
Expected: cuda available = True, device name contains "3090", memory ~24 GB.
"""
import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    total = torch.cuda.get_device_properties(0).total_memory // 1024**3
    print("vram GB:", total)
    # tiny smoke test on the GPU
    x = torch.randn(1000, 1000, device="cuda")
    print("smoke test ok:", (x @ x).shape)
else:
    print("GPU not visible - check NVIDIA driver and that torch was installed")
    print("from the CUDA index (see bio-ai-start-plan, Phase 0, step 3).")