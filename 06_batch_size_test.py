import gc
import csv
from pathlib import Path
import torch

def mib(bytes_value):
	return bytes_value / 1024**2
def reset_memory():
	gc.collect()
	torch.cuda.empty_cache()
	torch.cuda.synchronize()

def get_memory():
	free_bytes, total_bytes = torch.cuda.mem_get_info()

	allocated = torch.cuda.memory_allocated()
	reserved = torch.cuda.memory_reserved()

	return{
		"allocated_mib": round(mib(allocated), 2),
		"reserved_mib:": round(mib(reserved), 2),
		"free_mib": round(mib(free_bytes), 2),
		"total_mib": round(mib(total_bytes), 2),
	}

results = []

seq_len = 2048
hidden_size = 4096
dtype = torch.float16

for batch_size in [1, 2, 4, 8, 16, 24, 32, 48, 64]:
	reset_memory()

	shape = (batch_size, seq_len, hidden_size)

	try:
		x = torch.empty(shape, device="cuda", dtype=dtype)
		y = x * 1.01
		torch.cuda.synchronize()

		mem = get_memory()

		row = {
			"batch_size": batch_size,
			"shape": str(shape),
			"dtype": "float16",
			**mem,
			"status": "ok",
		}

		results.append(row)

		print(row)

		del x,y

	except torch.cuda.OutOfMemoryError:
		mem = get_memory()

		row = {
			"batch_size": batch_size,
			"shape": str(shape),
			"dtype": "float16",
			**mem,
			"status": "oom",
		}

		results.append(row)

		print(row)
		break
reset_memory()

Path("results").mkdir(exist_ok=True)

with open ("results/batch_size_results.csv", "w", newline="") as f:
	writer = csv.DictWriter(f, fieldnames = results[0].keys())
	writer.writeheader()
	writer.writerows(results)

print("\n Saved results/batch_size_results.csv")
