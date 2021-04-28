import torch
CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")

print(device)