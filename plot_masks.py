import numpy as np
import matplotlib.pyplot as plt
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "P01_IMG1.npy"

# Incarca datele
data = np.load(path, allow_pickle=True)
if data.shape == () and data.dtype == object:
    data = data.item()

# Extrage imagine si masuri
image = data['image']
structures = data['structures']

# Creeaza figura cu subploturi 2x2
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Segmentation Masks', fontsize=16)

# Plot fiecare masca in negru-alb
structure_names = list(structures.keys())

# Creeaza figura cu subploturi 2x2
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Segmentation Masks', fontsize=16)
fig.patch.set_facecolor('#f0f0f0')  # Fundal gri pentru figura

# Plot fiecare masca in negru-alb
structure_names = list(structures.keys())
colors_bg = ['#e8f4f8', '#f4e8e8', '#f0f4e8', '#f8e8f4']  # Culori pastel diferite

for idx, name in enumerate(structure_names):
    row = idx // 2
    col = idx % 2
    
    mask = structures[name]
    # Inverseaza masca pentru a fi mai vizibila (negru pe alb)
    axes[row, col].imshow(1 - mask, cmap='gray')
    axes[row, col].set_title(name.capitalize(), fontsize=14, fontweight='bold')
    axes[row, col].axis('off')
    
    # Adauga fundal colorat pe fiecare subplot
    axes[row, col].set_facecolor(colors_bg[idx])
    # Adauga border colorat
    for spine in axes[row, col].spines.values():
        spine.set_edgecolor(plt.cm.Set3(idx))
        spine.set_linewidth(4)
        spine.set_visible(True)

plt.tight_layout()
plt.savefig('segmentation_masks.png', dpi=150, bbox_inches='tight')
print("✓ Imaginea a fost salvata ca 'segmentation_masks.png'")
plt.show()
