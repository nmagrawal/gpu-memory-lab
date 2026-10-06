import torch

print("PyTorch version:", torch.__version__)
print("CUDA available", torch.cuda.is_available())

if torch.cuda.is_available():
	print("GPU name:", torch.cuda.get_device_name(0))
	props = torch.cuda.get_device_properties(0)
	print("Total VRM GB:", round(props.total_memory / 1024**3, 2))
else:
	print("CUDA is not available")
