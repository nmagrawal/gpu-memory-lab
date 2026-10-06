import torch

def mib(bytes_value):
	return bytes_value / 1024**2

def print_gpu_memory(label):
	free_bytes, total_bytes = torch.cuda.mem_get_info()

	allocated = torch.cuda.memory_allocated()
	reserved = torch.cuda.memory_reserved()

	print(f"\n{label}")
	print("-"*40)
	print("Allocated Mib:", round(mib(allocated), 2))
	print("Reserved MiB: ", round(mib(reserved), 2))
	print("Free MiB:     ", round(mib(free_bytes), 2))
	print("Total MiB:    ", round(mib(total_bytes), 2))

print_gpu_memory("Initial GPU memory")
