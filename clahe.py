
PATH_IN  = r"C:\Users\AlexVisoiu\facultate\dataset_clean"
PATH_OUT = r"C:\Users\AlexVisoiu\facultate\dataset_clahe"

import numpy as np
import cv2
import os
import sys

# Dimensiunea tinta pentru resize
TARGET_SIZE = (256, 256)  # (W, H) pentru cv2.resize

# Parametri CLAHE — ajusteaza dupa nevoie
CLIP_LIMIT     = 2.0    # contrast limit; mai mare = contrast mai agresiv
TILE_GRID_SIZE = (8, 8) # dimensiunea grilei de tile-uri


def to_grayscale_2d(image: np.ndarray) -> np.ndarray:
    """Normalizeaza orice imagine la un array 2D grayscale float32 in [0,1].
    Gestioneaza: (H,W), (H,W,1), (H,W,3), (H,W,4)."""
    img = np.squeeze(image)  # elimina dimensiunile de marime 1

    if img.ndim == 2:
        pass  # deja 2D
    elif img.ndim == 3:
        channels = img.shape[2]
        if channels == 3:
            # Converteste RGB -> grayscale (canale probabil identice, dar facem corect)
            img_u8 = to_uint8(img)
            img = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        elif channels == 4:
            img_u8 = to_uint8(img[:, :, :3])
            img = cv2.cvtColor(img_u8, cv2.COLOR_RGBA2GRAY).astype(np.float32) / 255.0
        else:
            raise ValueError(f"Numar de canale neasteptat: {channels}")
    else:
        raise ValueError(f"Shape neasteptat: {image.shape}")

    # Normalizeaza la float32 [0,1]
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    return img


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Converteste un array la uint8 [0,255] indiferent de dtype."""
    arr = arr.astype(np.float64)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


def resize_image(image: np.ndarray) -> np.ndarray:
    """Redimensioneaza o imagine 2D grayscale la TARGET_SIZE."""
    if image.shape == TARGET_SIZE:
        return image
    return cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)


def resize_mask(mask: np.ndarray) -> np.ndarray:
    """Redimensioneaza o masca binara la TARGET_SIZE cu INTER_NEAREST."""
    msk = np.squeeze(mask)
    if msk.ndim != 2:
        raise ValueError(f"Shape neasteptat masca: {msk.shape}")
    if msk.shape == TARGET_SIZE:
        return mask.astype(np.float32)
    resized = cv2.resize(msk.astype(np.float32), TARGET_SIZE,
                         interpolation=cv2.INTER_NEAREST)
    return resized


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Aplica CLAHE pe o imagine 2D float32 [0,1].
    Returneaza float32 [0,1]."""
    img_u8 = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    enhanced = clahe.apply(img_u8)
    return enhanced.astype(np.float32) / 255.0


def process_file(src_path: str, dst_path: str) -> None:
    data = np.load(src_path, allow_pickle=True)
    if data.shape == () and data.dtype == object:
        data = data.item()

    image      = data['image']
    structures = data['structures']

    # 1. Normalizeaza la grayscale 2D float32
    image_gray = to_grayscale_2d(image)

    # 2. Resize imagine
    image_resized = resize_image(image_gray)

    # 3. Resize masti
    structures_resized = {
        name: resize_mask(mask)
        for name, mask in structures.items()
    }

    # 4. CLAHE
    image_final = apply_clahe(image_resized)

    result = {
        'image':      image_final,
        'structures': structures_resized,
    }

    np.save(dst_path, result)


def main():
    if not PATH_IN or not PATH_OUT:
        print("Eroare: PATH_IN si PATH_OUT nu sunt setate. Editeaza scriptul.")
        sys.exit(1)

    if not os.path.isdir(PATH_IN):
        print(f"Eroare: PATH_IN nu exista sau nu e un folder: {PATH_IN}")
        sys.exit(1)

    os.makedirs(PATH_OUT, exist_ok=True)

    files = sorted(f for f in os.listdir(PATH_IN) if f.endswith('.npy'))

    if not files:
        print(f"Niciun fisier .npy gasit in: {PATH_IN}")
        sys.exit(0)

    print(f"Gasit {len(files)} fisiere .npy in {PATH_IN}")
    print(f"Output -> {PATH_OUT}")
    print(f"Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}, CLAHE clip={CLIP_LIMIT}\n")

    for i, fname in enumerate(files, 1):
        src = os.path.join(PATH_IN,  fname)
        dst = os.path.join(PATH_OUT, fname)
        try:
            process_file(src, dst)
            print(f"[{i:>4}/{len(files)}] OK  {fname}")
        except Exception as e:
            print(f"[{i:>4}/{len(files)}] ERR {fname}: {e}")

    print("\nGata.")


if __name__ == "__main__":
    main()