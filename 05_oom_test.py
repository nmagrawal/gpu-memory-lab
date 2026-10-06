import gc
import torch

torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(0.90, device=0)


def mib(bytes_value):
	return bytes_value / 1024 **2

def reset_memory():
	gc.collect()
	torch.cuda.empty_cache()
	torch.cuda.synchronize()

def print_gpu_memory(label):
	free_bytes, total_bytes = torch.cuda.mem_get_info()

	print(f"{label:10s} |"
		f"allocated= {mib(torch.cuda.memory_allocated()):8.2f} MiB | "
		f"reserved={mib(torch.cuda.memory_reserved()):8.2f} MiB | "
		f"free= {mib(free_bytes):8.2f} MiB")

reset_memory()

tensors = []
last_success = None

try:
	for n in range(4096, 70000, 4096):
		shape = (n, n)

		x = torch.ones(shape, device="cuda", dtype=torch.float32)
		torch.cuda.synchronize()

		tensors.append(x)
		last_success = shape

		print_gpu_memory(f"Success {shape}")

except torch.cuda.OutOfMemoryError:
	print("\n CUDA out of memory happned.")
	print("Last successful shape:", last_success)
	print_gpu_memory("At OOM")

finally:
	del tensors
	reset_memory()
	print_gpu_memory("After cleanup")
