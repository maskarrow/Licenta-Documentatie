import os
import glob
import re
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Config:
    DATA_DIR = '/content/dataset_local'

    IN_CHANNELS = 1
    BATCH_SIZE = 8
    NUM_EPOCHS = 150
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    MAX_GRAD_NORM = 1.0
    PATIENCE_EARLY_STOPPING = 150

    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 2
    PIN_MEMORY = True


# Arhitectura modelului

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class MultiOutputUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        rgb_w = backbone.conv1.weight.clone()
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1.weight.data = torch.sum(rgb_w, dim=1, keepdim=True)
        self.enc1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc2 = backbone.layer1
        self.enc3 = backbone.layer2
        self.enc4 = backbone.layer3
        self.enc5 = backbone.layer4
        self.dec4 = DecoderBlock(2048, 1024, 512)
        self.dec3 = DecoderBlock(512, 512, 256)
        self.dec2 = DecoderBlock(256, 256, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        a1 = self.enc1(x)
        a2 = self.enc2(self.pool(a1))
        a3 = self.enc3(a2)
        a4 = self.enc4(a3)
        a5 = self.enc5(a4)
        b4 = self.dec4(a5, a4)
        b3 = self.dec3(b4, a3)
        b2 = self.dec2(b3, a2)
        b1 = self.dec1(b2, a1)
        return self.final_conv(b1)


# Funcția de pierdere și metricile

class MultiOrganLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        pos_weights = torch.tensor([150.0, 15.0, 20.0, 30.0]).view(1, -1, 1, 1).to(Config.DEVICE)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        self.alpha = 0.4
        self.beta = 0.6
        self.gamma = 0.75

    def forward(self, logits, targets):
        loss_bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-5
        tp = (probs * targets).sum(dim=(2, 3))
        fp = ((1 - targets) * probs).sum(dim=(2, 3))
        fn = (targets * (1 - probs)).sum(dim=(2, 3))
        tversky = (tp + smooth) / (tp + self.alpha * fn + self.beta * fp + smooth)
        tversky_loss = torch.pow((1 - tversky), self.gamma).mean()
        return 0.4 * loss_bce + 0.6 * tversky_loss

def compute_metrics(logits, targets, organ_names):
    preds = (torch.sigmoid(logits) > 0.5).float()
    metrics = {organ: {} for organ in organ_names}

    tp = (preds * targets).sum(dim=(0, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(0, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(0, 2, 3))
    tn = ((1 - preds) * (1 - targets)).sum(dim=(0, 2, 3))

    smooth = 1e-7

    dsc  = (2 * tp) / (2 * tp + fp + fn + smooth)
    jac  = tp / (tp + fp + fn + smooth)
    sens = tp / (tp + fn + smooth)
    spec = tn / (tn + fp + smooth)
    prec = tp / (tp + fp + smooth)

    for i, organ in enumerate(organ_names):
        metrics[organ]['dice'] = dsc[i].item()
        metrics[organ]['iou']  = jac[i].item()
        metrics[organ]['sens'] = sens[i].item()
        metrics[organ]['spec'] = spec[i].item()
        metrics[organ]['prec'] = prec[i].item()

    return metrics


# Setul de date

def analyze_and_split_dataset(data_dir, val_ratio=0.15, test_ratio=0.15, seed=Config.SEED):
    all_files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
    if not all_files:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")
    sample = np.load(all_files[0], allow_pickle=True).item()
    organ_names = sorted(sample['structures'].keys())

    pat_ids = sorted(set(re.search(r'(P\d+)_IMG', os.path.basename(f)).group(1) for f in all_files))
    tr_pats, tmp_pats = train_test_split(pat_ids, test_size=val_ratio + test_ratio, random_state=seed)
    vl_pats, ts_pats  = train_test_split(tmp_pats, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)

    def filter_by_patient(p_set):
        return [f for f in all_files if re.search(r'(P\d+)_IMG', os.path.basename(f)).group(1) in p_set]

    train_paths = filter_by_patient(set(tr_pats))
    val_paths   = filter_by_patient(set(vl_pats))
    test_paths  = filter_by_patient(set(ts_pats))

    print(f"Organs: {organ_names}")
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
    return train_paths, val_paths, test_paths, organ_names

class MultiOrganDataset(Dataset):
    def __init__(self, file_paths, organ_names, transform=None):
        self.paths = file_paths
        self.organ_names = organ_names
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx], allow_pickle=True).item()

        image = np.ascontiguousarray(data['image'].astype(np.float32))
        mask_list = [np.ascontiguousarray(data['structures'][organ].astype(np.float32)) for organ in self.organ_names]

        if self.transform:
            augmented = self.transform(image=image, masks=mask_list)
            image = augmented['image']
            mask_list = augmented['masks']

            masks = torch.stack(
                [m if isinstance(m, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(m)) for m in mask_list]
            ).float()
        else:
            masks = torch.stack([torch.from_numpy(m) for m in mask_list]).float()

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.ascontiguousarray(image))

        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = image.permute(2, 0, 1)

        return {'image': image.float(), 'masks': masks}


# Augmentări și pregătirea antrenării

def get_transforms():
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.2),
        A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.2),
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2()
    ], is_check_shapes=False)

    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2()
    ], is_check_shapes=False)

    return train_transform, val_transform


train_paths, val_paths, test_paths, organ_names = analyze_and_split_dataset(Config.DATA_DIR)
num_classes = len(organ_names)

train_transform, val_transform = get_transforms()

train_ds = MultiOrganDataset(train_paths, organ_names, train_transform)
val_ds   = MultiOrganDataset(val_paths,   organ_names, val_transform)

train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
                          num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE, shuffle=False,
                          num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

model     = MultiOutputUNet(num_classes).to(Config.DEVICE)
criterion = MultiOrganLoss(num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
scaler    = GradScaler('cuda')

best_dice  = 0.0
no_improve = 0


# Bucla de antrenare

for epoch in range(1, Config.NUM_EPOCHS + 1):

    model.train()
    train_loss = 0.0
    bar_train = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.NUM_EPOCHS} [Train]", leave=False)

    for batch in bar_train:
        images, masks = batch['image'].to(Config.DEVICE), batch['masks'].to(Config.DEVICE)

        optimizer.zero_grad()
        with autocast(Config.DEVICE):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        bar_train.set_postfix_str(f"Loss: {loss.item():.4f}")

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    val_stats = {organ: {'dice': [], 'iou': [], 'sens': [], 'spec': [], 'prec': []} for organ in organ_names}
    bar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{Config.NUM_EPOCHS} [Val  ]", leave=False)

    with torch.no_grad():
        for batch in bar_val:
            images, masks = batch['image'].to(Config.DEVICE), batch['masks'].to(Config.DEVICE)

            with autocast(Config.DEVICE):
                logits = model(images)
                loss = criterion(logits, masks)

            val_loss += loss.item()
            metrics = compute_metrics(logits, masks, organ_names)

            for organ in organ_names:
                for k in val_stats[organ]:
                    val_stats[organ][k].append(metrics[organ][k])

    val_loss /= len(val_loader)

    dice_per_organ = []
    print(f"\n{'-'*95}")
    print(f"EPOCH {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"{'-'*95}")
    print(f"{'Organ':<15} | {'Dice':<8} | {'IoU':<8} | {'Sens':<8} | {'Spec':<8} | {'Prec':<8}")
    print(f"{'-'*95}")

    for organ in organ_names:
        d  = np.mean(val_stats[organ]['dice'])
        i  = np.mean(val_stats[organ]['iou'])
        sn = np.mean(val_stats[organ]['sens'])
        sp = np.mean(val_stats[organ]['spec'])
        pr = np.mean(val_stats[organ]['prec'])
        dice_per_organ.append(d)
        print(f"{organ:<15} | {d:.4f}   | {i:.4f}   | {sn:.4f}   | {sp:.4f}   | {pr:.4f}")

    avg_val_dice = np.mean(dice_per_organ)
    print(f"{'-'*95}")
    print(f"MACRO AVG       | {avg_val_dice:.4f}")
    print(f"{'-'*95}\n")

    scheduler.step()

    if avg_val_dice > best_dice:
        best_dice  = avg_val_dice
        no_improve = 0
        torch.save({'model_state_dict': model.state_dict()}, 'best_model.pt')
        print("New best model saved.")
    else:
        no_improve += 1

    if no_improve >= Config.PATIENCE_EARLY_STOPPING:
        print("Early stopping triggered.")
        break

print("\nTraining complete.")
