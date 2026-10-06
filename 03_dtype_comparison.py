import gc
import torch

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

shape = (12000,12000)

dtypes = [
	torch.float32,
	torch.float16,
	torch.bfloat16,
]

for dtype in dtypes:
	reset_memory()

	x = torch.empty(shape, device="cuda", dtype=dtype)
	torch.cuda.synchronize()

	print_gpu_memory(str(dtype).replace("torch.", ""))

	del x
	reset_memory()
