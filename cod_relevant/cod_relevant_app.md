# Părți relevante din app_curat.py pentru lucrarea de licență

Codul implementează o **aplicație desktop de inferență** care încarcă modelul antrenat și rulează segmentarea pe imagini CT noi. Mai jos sunt explicate secțiunile cu valoare academică și motivul pentru care merită (sau nu) incluse în anexă.

---

## Ce SE include și de ce

### 1. Pipeline-ul de preprocesare — `filtru_mediu`, `dilate_mask`, `erode_mask`, `curata_zgomot`, `elimina_suprapunere_text`, `preprocesare`

```python
def filtru_mediu(img, raza=3): ...
def dilate_mask(mask, raza=1): ...
def erode_mask(mask, raza=1): ...
def curata_zgomot(mask, raza=1): ...
def elimina_suprapunere_text(img): ...
def preprocesare(img): ...
```

**De ce e relevant:**
- Acesta este singurul bloc de cod cu adevărat specific domeniului din aplicație — o contribuție metodologică proprie, nu un wrapper UI.
- `filtru_mediu` implementează un **filtru box prin imagine integrală** (summed-area table), evitând buclele pixel-cu-pixel. Tehnica merită menționată explicit.
- `elimina_suprapunere_text` detectează și elimină adnotările text arse pe imaginile CT (artefacte instituționale comune în seturile de date reale), folosind deviația față de media locală ca detector de anomalii.
- `dilate_mask` / `erode_mask` implementează **operații morfologice** de bază cu NumPy pur (fără OpenCV), prin fereastră glisantă cu padding.
- `curata_zgomot` aplică o **deschidere morfologică** (erode → dilate) pentru eliminarea zgomotului punctiform din masca de text.
- `preprocesare` combină curățarea textului cu **CLAHE** (Contrast Limited Adaptive Histogram Equalization) — aceeași normalizare aplicată la generarea dataset-ului de antrenare.

**Ce explici în text:** necesitatea eliminării artefactelor text din imaginile CT clinice, rolul CLAHE în uniformizarea contrastului local, operațiile morfologice de curățare.

---

### 2. Funcția de inferență — `inferenta`

```python
def inferenta(model, img2d, device):
    aug = aug_val(image=img2d)
    t = aug['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(t)
        pred = (torch.sigmoid(logits) > 0.5).float()
    return pred[0].cpu().numpy()
```

**De ce e relevant:**
- Ilustrează complet fluxul de inferență: normalizare → tensor → forward pass → sigmoid → prag binar.
- `torch.no_grad()` dezactivează gradienții la inferență — decizie de eficiență standard care merită menționată.
- Pragul `0.5` pe sigmoid este consistent cu cel din antrenare (`compute_metrics`).
- `unsqueeze(0)` adaugă dimensiunea de batch pentru un singur exemplu.

**Ce explici în text:** fluxul complet de la imagine raw la mască binară, consistența preprocesării între antrenare și inferență.

---

### 3. Transformarea de inferență — `aug_val`

```python
aug_val = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
    ToTensorV2()
])
```

**De ce e relevant:**
- Demonstrează că la inferență se aplică **doar normalizarea**, fără nicio augmentare — identic cu `val_transform` din antrenare.
- Consistența între preprocesarea la antrenare și inferență este o condiție necesară pentru rezultate corecte. Merită subliniată explicit.

---

### 4. Încărcarea modelului — `incarca_model`

```python
def incarca_model(device):
    net = SegModel(NR_CLASE).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    net.load_state_dict(ckpt)
    net.eval()
    return net
```

**De ce e relevant:**
- Arată cum se restaurează un model antrenat din fișier `.pt`: instanțiere arhitectură → `load_state_dict` → `eval()`.
- `map_location=device` permite rularea pe CPU dacă GPU nu este disponibil — portabilitate.
- `net.eval()` comută BatchNorm și Dropout în modul de inferență (dezactivează comportamentul stochastic).

---

### 5. Arhitectura modelului (versiunea de inferență) — `UpBlock` + `SegModel`

```python
class UpBlock(nn.Module): ...
class SegModel(nn.Module): ...
```

**De ce e relevant (cu rezervă):**
- Arhitectura este **identică funcțional** cu `DecoderBlock` + `MultiOutputUNet` din `train_curat.py`. Diferența: `weights=None` (nu se mai încarcă ImageNet — ponderile vin din checkpoint-ul salvat) și denumirile interne diferă (`s1–s5`, `up1–up4`, `cap` vs `enc1–enc5`, `dec1–dec4`, `final_conv`).
- **Recomandare:** nu duplica codul în anexă. Menționează în text că definiția arhitecturii este identică cu cea din antrenare, cu singura diferență că `weights=None` la construire.

---

### 6. Citirea datelor — `citeste_imagine_npy`

```python
def citeste_imagine_npy(cale):
    date = np.load(cale, allow_pickle=True)
    ...
    return img  # array 2D float32
```

**De ce e relevant (opțional):**
- Documentează formatul fișierelor `.npy`: pot fi dict cu cheia `'image'` sau array direct.
- Gestionează cazuri de edge: array 3D cu 3 canale (ia primul canal), shape `()` de tip object.
- **Recomandare:** include dacă discuți formatul datelor în subcapitolul de date; altfel, poate fi omis.

---

## Ce NU se include și de ce

| Secțiune | Motiv excludere |
|---|---|
| Importuri `tkinter`, `customtkinter`, `matplotlib.backends.backend_tkagg`, `matplotlib.patches` | Biblioteci exclusiv de interfață grafică |
| Constante de culoare (`BG_MAIN`, `CLR_ACCENT`, etc.) | Configurare vizuală, fără relevanță metodologică |
| `COLORS`, `COLORS_RGB`, `hex_to_float` | Culori de afișare per organ — detaliu UI |
| `ctk.set_appearance_mode`, `ctk.set_default_color_theme` | Configurare temă UI |
| Clasa `App` în întregime | Interfață grafică — zero conținut metodologic |
| `_bara_instrumente`, `_legenda`, `_panouri`, `_creeaza_panou`, `_bara_status` | Construcție widget-uri UI |
| `_redeseaza_stanga`, `_redeseaza_dreapta` | Redare matplotlib în fereastră Tkinter |
| `_comuta_overlay`, `_comuta_organ`, `_schimba_opacitate` | Interactivitate UI |
| `_afiseaza_incarcare` | Animație bară de progres |
| `_ruleaza_segmentare` (întreg) | Wrapper UI pentru `preprocesare` + `inferenta`; logica de threading este detaliu de implementare, nu contribuție |
| `_initializeaza_model` | Verificare existență fișier + messagebox — infrastructură |
| `_deschide_fisier` | Dialog selectare fișier — UI |
| `SCRIPT_DIR`, `MODEL_PATH` | Cale fișier specifică instalației locale |
| `norm_to_uint8` | Utilitar intern folosit în `elimina_suprapunere_text`; poate fi inclus alături de aceasta sau omis dacă codul e parafrazat |
| `if __name__ == "__main__"` | Punct de intrare al aplicației |

---

## Recomandare de structurare în lucrare

1. **Subcapitol Preprocesare date la inferență**: include `filtru_mediu` + `elimina_suprapunere_text` + `preprocesare` cu explicația detaliată a detecției artefactelor text și a CLAHE
2. **Subcapitol Inferență**: include `aug_val` + `inferenta` cu fluxul complet imagine → mască
3. **Subcapitol Restaurare model**: include `incarca_model` cu explicarea `eval()` și `map_location`
4. **Mențiune în text** (fără cod duplicat): arhitectura `SegModel` este identică cu `MultiOutputUNet` din antrenare, cu `weights=None` la inițializare
