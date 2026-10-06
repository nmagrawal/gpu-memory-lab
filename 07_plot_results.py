from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

Path("plots").mkdir(exist_ok=True)

df = pd.read_csv("results/batch_size_results.csv")

df_ok = df[df["status"] == "ok"]

plt.figure()
plt.plot(df_ok["batch_size"], df_ok ["allocated_mib"], marker="o")
plt.xlabel("Batch size")
plt.ylabel("Allocated VRAM MiB")
plt.title("Batch size vs allocated GPU memory")
plt.grid(True)
plt.savefig("plots/batch_size_vs_vram.png", dpi=150, bbox_inches="tight")

print("Saved plots/batch_size_vs_vram.png")
