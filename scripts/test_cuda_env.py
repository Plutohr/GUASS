import torch
import torchvision
import diffusers
import transformers
import PIL
import yaml
import numpy

print("torch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)
print("diffusers version:", diffusers.__version__)
print("transformers version:", transformers.__version__)
print("Pillow version:", PIL.__version__)
print("PyYAML version:", yaml.__version__)
print("numpy version:", numpy.__version__)

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    x = torch.randn(2, 3, device="cuda")
    y = torch.randn(2, 3, device="cuda")
    z = x + y
    print("CUDA test tensor device:", z.device)
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available")
