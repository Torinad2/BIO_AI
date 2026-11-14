"""
RU: check_env.py — проверка Фазы 0: убеждаемся, что Python видит RTX 3090 и её ~24 ГБ.
    Запускать внутри conda-окружения `bioai`:
        python check_env.py
    Ожидается: cuda available = True, имя устройства содержит "3090", память ~24 ГБ.
EN: check_env.py — Phase 0 check: confirm Python sees the RTX 3090 and its ~24 GB.
    Run inside the `bioai` conda environment:
        python check_env.py
    Expected: cuda available = True, device name contains "3090", memory ~24 GB.
"""
#RU: PyTorch: версия, доступность CUDA и вычисления на GPU
#EN: PyTorch: version, CUDA availability and GPU computation
import torch

#RU: Версия установленного PyTorch
#EN: Installed PyTorch version
print("torch version:", torch.__version__)

#RU: Видит ли PyTorch CUDA (должно быть True, если драйвер и сборка torch подходят)
#EN: Whether PyTorch can see CUDA (should be True if the driver and the torch build match)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    #RU: Имя первой видеокарты (индекс 0)
    #EN: Name of the first GPU (index 0)
    print("device:", torch.cuda.get_device_name(0))

    #RU: Общий объём видеопамяти в ГБ (байты // 1024**3)
    #EN: Total video memory in GB (bytes // 1024**3)
    total = torch.cuda.get_device_properties(0).total_memory // 1024**3
    print("vram GB:", total)

    #RU: Небольшой тест на GPU: умножение матриц 1000x1000 прямо на видеокарте
    #EN: Tiny smoke test on the GPU: a 1000x1000 matrix multiplication on the card itself
    x = torch.randn(1000, 1000, device="cuda")
    print("smoke test ok:", (x @ x).shape)
else:
    #RU: GPU не виден: проверьте драйвер NVIDIA и то, что torch установлен из индекса CUDA
    #EN: GPU not visible: check the NVIDIA driver and that torch was installed from the CUDA index
    print("GPU not visible - check NVIDIA driver and that torch was installed")
    print("from the CUDA index (see bio-ai-start-plan, Phase 0, step 3).")