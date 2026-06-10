import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASE    = ['artery', 'liver', 'stomach', 'vein']
NR_CLASE = 4

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Transformare de inferenta: doar normalizare, fara augmentari (identic cu val_transform din antrenare)
aug_val = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
    ToTensorV2()
])


# Arhitectura modelului — versiunea de inferenta (identica cu MultiOutputUNet din antrenare, weights=None)

class UpBlock(nn.Module):
    def __init__(self, ch_in, ch_skip, ch_out):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        total_in = ch_in + ch_skip
        self.convblock = nn.Sequential(
            nn.Conv2d(total_in, ch_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.convblock(x)

class SegModel(nn.Module):
    def __init__(self, nr_clase):
        super().__init__()
        backbone = models.resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.s1   = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.s2   = backbone.layer1
        self.s3   = backbone.layer2
        self.s4   = backbone.layer3
        self.s5   = backbone.layer4

        self.up4 = UpBlock(2048, 1024, 512)
        self.up3 = UpBlock(512,  512,  256)
        self.up2 = UpBlock(256,  256,  128)
        self.up1 = UpBlock(128,  64,   64)

        self.cap = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, nr_clase, kernel_size=1)
        )

    def forward(self, x):
        f1 = self.s1(x)
        f2 = self.s2(self.pool(f1))
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        f5 = self.s5(f4)
        d4 = self.up4(f5, f4)
        d3 = self.up3(d4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)
        return self.cap(d1)


# Pipeline de preprocesare

def norm_to_uint8(img):
    if img.dtype == np.uint8:
        return img
    tmp = img.astype(np.float32)
    tmp = tmp - tmp.min()
    mx = tmp.max()
    if mx > 0:
        tmp = tmp / mx * 255.0
    return np.clip(tmp, 0, 255).astype(np.uint8)

def filtru_mediu(img, raza=3):
    img = img.astype(np.float32)
    bordura = np.pad(img, raza, mode="reflect")
    H, W = img.shape
    dim = 2 * raza + 1

    # imagine integrala pentru media locala eficienta
    integ = np.zeros((bordura.shape[0] + 1, bordura.shape[1] + 1), dtype=np.float32)
    integ[1:, 1:] = bordura.cumsum(axis=0).cumsum(axis=1)

    a = integ[dim : dim+H, dim : dim+W]
    b = integ[:H,  dim : dim+W]
    c = integ[dim : dim+H, :W]
    d = integ[:H,  :W]
    return (a - b - c + d) / (dim * dim)

def dilate_mask(mask, raza=1):
    mask = mask.astype(bool)
    tmp = np.pad(mask, raza, mode="constant", constant_values=False)
    rezultat = np.zeros_like(mask)
    for dy in range(2*raza + 1):
        for dx in range(2*raza + 1):
            rezultat |= tmp[dy : dy+mask.shape[0], dx : dx+mask.shape[1]]
    return rezultat

def erode_mask(mask, raza=1):
    mask = mask.astype(bool)
    tmp = np.pad(mask, raza, mode="constant", constant_values=False)
    rezultat = np.ones_like(mask)
    for dy in range(2*raza + 1):
        for dx in range(2*raza + 1):
            rezultat &= tmp[dy : dy+mask.shape[0], dx : dx+mask.shape[1]]
    return rezultat

def curata_zgomot(mask, raza=1):
    return dilate_mask(erode_mask(mask, raza), raza)

def elimina_suprapunere_text(img):
    img = norm_to_uint8(img)
    medie_loc = filtru_mediu(img, raza=10)
    masca_text = (img >= medie_loc + 22) | (img <= medie_loc - 22)
    masca_text = curata_zgomot(masca_text, raza=2)
    masca_text = dilate_mask(masca_text, raza=3)
    fundal = filtru_mediu(img, raza=12).astype(np.uint8)
    curat = img.copy()
    curat[masca_text] = fundal[masca_text]
    return curat

def preprocesare(img):
    curat = elimina_suprapunere_text(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(curat)


# Citirea datelor

def citeste_imagine_npy(cale):
    date = np.load(cale, allow_pickle=True)
    if isinstance(date, np.ndarray) and date.shape == () and date.dtype == object:
        date = date.item()
    if isinstance(date, dict):
        if 'image' not in date:
            raise ValueError("Dict-ul .npy nu contine cheia 'image'.")
        img = date['image']
    else:
        img = date
    img = np.asarray(img).astype(np.float32)
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img[:, :, 0]
    if img.ndim == 3:
        img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"Imaginea trebuie sa fie 2D dupa procesare, am primit shape {img.shape}.")
    return img


# Incarcarea modelului

def incarca_model(model_path, device):
    net = SegModel(NR_CLASE).to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    net.load_state_dict(ckpt)
    net.eval()
    return net


# Inferenta

def inferenta(model, img2d, device):
    aug = aug_val(image=img2d)
    t = aug['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(t)
        pred = (torch.sigmoid(logits) > 0.5).float()
    return pred[0].cpu().numpy()
