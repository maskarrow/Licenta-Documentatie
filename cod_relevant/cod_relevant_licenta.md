# Părți relevante din cod pentru lucrarea de licență

Codul implementează un sistem de **segmentare automată multi-organ** din imagini medicale (CT/RMN), folosind o arhitectură U-Net cu backbone ResNet50 preantrenat. Mai jos sunt explicate secțiunile cu valoare academică și motivul pentru care merită incluse în lucrare.

---

## Ce SE include și de ce

### 1. Arhitectura modelului — `DecoderBlock` + `MultiOutputUNet` (Blocul 3)

```python
class DecoderBlock(nn.Module): ...
class MultiOutputUNet(nn.Module): ...
```

**De ce e relevant:**
- Aceasta este contribuția centrală a lucrării — arhitectura efectivă.
- `MultiOutputUNet` adaptează ResNet50 (antrenat pe ImageNet) ca **encoder** pentru imagini cu un singur canal (grayscale), printr-un truc simplu: ponderile RGB sunt sumate pe canalul de culoare (`torch.sum(rgb_w, dim=1, keepdim=True)`). Merită explicat explicit în lucrare.
- `DecoderBlock` implementează **skip connections** caracteristice U-Net: upsample bilinear → concatenare cu feature map de la encoder → două convoluții cu BN și ReLU.
- Capul final (`final_conv`) produce câte o hartă de probabilitate per organ (multi-label, nu multi-class).

**Ce explici în text:** diagrama U-Net encoder–decoder, rolul skip connections, motivul folosirii ResNet50 preantrenat (transfer learning pe date medicale limitate).

---

### 2. Funcția de pierdere — `MultiOrganLoss` (Blocul 4)

```python
class MultiOrganLoss(nn.Module): ...
```

**De ce e relevant:**
- Reprezintă o **alegere de design motivată academic**: simpla entropie încrucișată (BCE) este insuficientă când organele mici ocupă <1% din imagine.
- Se folosește o combinație ponderată: `0.4 × BCE + 0.6 × Tversky Loss`.
- `pos_weight=[150, 15, 20, 30]` compensează dezechilibrul masiv de clase (fundal vs. organ).
- **Tversky Loss** cu `α=0.4, β=0.6` penalizează mai mult falsele negative (organul omis) față de falsele pozitive — esențial în context medical.
- Parametrul `γ=0.75` din `torch.pow((1 - tversky), γ)` aduce comportamentul Focal Tversky, focusând antrenarea pe cazuri dificile.

**Ce explici în text:** dezechilibrul de clase în imaginile medicale, Dice Loss vs Tversky Loss, alegerea parametrilor α/β, justificarea combinării celor două pierderi.

---

### 3. Metricile de evaluare — `compute_metrics` (Blocul 4)

```python
def compute_metrics(logits, targets, organ_names): ...
```

**De ce e relevant:**
- Calculează pe GPU, vectorizat pe toate organele simultan: **Dice, IoU (Jaccard), Sensitivitate, Specificitate, Precizie**.
- Aceste metrici sunt standardul în literatura de segmentare medicală — Dice este metrica principală de comparație cu alte lucrări.
- Alegerea pragului 0.5 pe probabilitățile sigmoid este o decizie de design care poate fi discutată.

**Ce explici în text:** definiția matematică a fiecărei metrici, de ce Dice (și nu Accuracy) este potrivit pentru segmentare medicală dezechilibrată.

---

### 4. Setul de date și încărcarea datelor — `MultiOrganDataset` + `analyze_and_split_dataset` (Blocul 5)

```python
def analyze_and_split_dataset(...): ...
class MultiOrganDataset(Dataset): ...
```

**De ce e relevant:**
- `analyze_and_split_dataset` face **split stratificat la nivel de pacient** (nu la nivel de slice), prevenind data leakage: toate feliile unui pacient merg în același subset (train/val/test). Aceasta este o decizie critică de corectitudine metodologică.
- `MultiOrganDataset` arată cum sunt structurate datele (fișiere `.npy` cu câmpurile `image` și `structures`), inclusiv conversia la tensori PyTorch și gestionarea transformărilor Albumentations.

**Ce explici în text:** importanța split-ului per pacient (nu per slice), structura fișierelor de date, cum sunt reprezentate măștile pentru fiecare organ.

---

### 5. Augmentările — `get_transforms` (Blocul 6)

```python
def get_transforms():
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(...),
        A.GaussNoise(p=0.2),
        A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.2),
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2()
    ])
```

**De ce e relevant:**
- Augmentările sunt o componentă activă a metodologiei, nu simplu boilerplate.
- `ElasticTransform` este specific domeniului medical — simulează deformările tisulare realiste.
- Normalizarea la `[−1, 1]` este adaptată pentru intrări grayscale.
- Val/test primesc **doar Resize + Normalize** (fără augmentări) — corect metodologic.

**Ce explici în text:** rolul augmentărilor în prevenirea overfitting-ului pe seturi medicale mici, alegerea specifică a `ElasticTransform` pentru imagini CT/RMN.

---

### 6. Hiperparametrii — clasa `Config` (Blocul 2, parțial)

```python
class Config:
    IN_CHANNELS = 1
    BATCH_SIZE = 8
    NUM_EPOCHS = 150
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    MAX_GRAD_NORM = 1.0
    PATIENCE_EARLY_STOPPING = 150
```

**De ce e relevant:**
- Tabelul de hiperparametri este standard în orice lucrare de ML — permite reproducibilitatea.
- `MAX_GRAD_NORM = 1.0` → gradient clipping, important pentru stabilitate cu Tversky.
- `PATIENCE_EARLY_STOPPING = 150` (egal cu NUM_EPOCHS) = practic dezactivat — o decizie conștientă care merită menționată.

---

### 7. Bucla de antrenare — secvența principală (Blocul 7, parțial)

```python
optimizer.zero_grad()
with autocast(Config.DEVICE):
    logits = model(images)
    loss = criterion(logits, masks)
scaler.scale(loss).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
scaler.step(optimizer)
scaler.update()
```

**De ce e relevant:**
- Folosirea **Mixed Precision Training** (AMP — `autocast` + `GradScaler`) — reduce memoria și accelerează antrenarea pe GPU, merită menționat.
- Ordinea corectă: `zero_grad → forward → backward → clip → step` este esențială și demonstrează înțelegerea procesului.
- `CosineAnnealingLR` ca scheduler — learning rate scade smooth, evitând oscilații la finalul antrenării.

---

## Ce NU se include și de ce

| Secțiune | Motiv excludere |
|---|---|
| Blocul 1 — montarea Google Drive | Infrastructură specifică Colab, fără relevanță academică |
| `setup_env()` | Boilerplate de mediu, nu parte din metodologie |
| Logica de resume training (`RESUME_TRAINING`) | Detaliu de implementare practică, nu contribuție metodologică |
| Salvarea checkpoint-urilor la epoci fixe (30, 60, 90...) | Detaliu operațional |
| `plot_results` (Blocul 9) | Cod auxiliar de vizualizare; **figurile** rezultate pot fi incluse, nu codul |
| Blocul 8 — testare | Se poate rezuma în text; codul în sine e repetitiv față de validare |

---

## Recomandare de structurare în lucrare

1. **Subcapitol Arhitectură**: include `DecoderBlock` + `MultiOutputUNet` cu diagramă U-Net
2. **Subcapitol Funcție de cost**: include `MultiOrganLoss` cu formula matematică explicată
3. **Subcapitol Metrici**: include `compute_metrics` sau doar formulele matematice
4. **Subcapitol Pregătirea datelor**: include `analyze_and_split_dataset` + tabel cu split train/val/test + `get_transforms`
5. **Subcapitol Antrenare**: include fragmentul AMP din bucla de antrenare + tabelul cu hiperparametri din `Config`
6. **Subcapitol Rezultate**: tabele cu Dice/IoU/Sens/Spec/Prec și figuri din `plot_results`
