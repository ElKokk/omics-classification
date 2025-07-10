from pathlib import Path
import matplotlib.pyplot as plt

def write_stub_png(png_path):
    """Create a 1×1 transparent PNG so Snakemake has something to touch."""
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(1,1))
    plt.axis('off')
    plt.savefig(png_path, dpi=72, bbox_inches='tight', transparent=True)
    plt.close()
