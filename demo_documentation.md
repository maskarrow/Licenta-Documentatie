# Documentație aplicație demo — Segmentare Organe Fetale

## Descriere generală

Aplicație desktop pentru inferența interactivă a modelului MultiOutputUNet.
Construită cu CustomTkinter (interfață grafică nativă Python), tema dark.
Fișier principal: `app.py`. Model necesar: `best_model.pt` în același director.
Rezoluție fereastră: 1280×820 px, dimensiune minimă 1200×760 px.
Titlul ferestrei: "Segmentare Organe".

---

## Stack tehnologic

| Bibliotecă | Rol |
|---|---|
| customtkinter | UI modern dark-themed |
| tkinter | filedialog, messagebox, frame de bază |
| matplotlib + FigureCanvasTkAgg | afișare imagini embedded în UI |
| OpenCV (cv2) | CLAHE |
| NumPy | procesare imagini |
| PyTorch | inferență model |
| Albumentations | normalizare tensor |
| threading | inferență non-blocantă |

---

## Layout interfață

### 1. Toolbar (sus)
- **Browse Image** — deschide file dialog, acceptă doar fișiere `.npy`
- **Segment** — pornește inferența (dezactivat până la încărcarea unei imagini)
- **Overlay View** — toggle suprapunere măști pe imaginea originală (dezactivat până după inferență)
- **Separator vizual**
- **Slider opacitate** — 0–100%, valoare implicită 70%; controlează transparența măștilor în overlay
- **Pill status** — indicator text/culoare al stării curente (niciun fișier / fișier încărcat / N organe segmentate)

### 2. Bara de organe (sub toolbar)
Patru chip-uri clickabile, câte unul per organ, cu swatch de culoare:
- **Artery** — amber `#f59e0b`
- **Liver** — roșu `#ef4444`
- **Stomach** — verde `#22c55e`
- **Vein** — albastru `#3b82f6`

Click pe un chip togglează vizibilitatea organului respectiv în ambele panouri. Când un organ e dezactivat, chip-ul devine estompat.

### 3. Panoul stâng — "Imagine originală"
Afișează imaginea ecografică brută (grayscale) după încărcare.
Când Overlay View este activ, afișează imaginea cu măștile suprapuse color cu opacitatea setată.
Legenda culorilor apare în colțul dreapta-sus al imaginii când overlay-ul e activ.
Header: dimensiunile imaginii (W × H) în colțul drept.

### 4. Panoul drept — "Măști segmentare"
Grilă 2×2 cu cele patru măști binare de segmentare, una per organ.
Titlul fiecărei subparcele este colorat în culoarea organului respectiv.
Dacă un organ e dezactivat, subparcela afișează o mască neagră.
În timpul inferenței, panoul drept ascunde conținutul și afișează o bară de loading indeterminată cu textul "Segmentare în curs…".
Header: numărul total de organe produse de model ("4 organe").

### 5. Status bar (jos)
- Stânga: informații despre fișierul curent (nume fișier, dimensiuni)
- Dreapta: numele fișierului model (`best_model.pt`) + indicatorul temei (`dark`)

---

## Pipeline inferență

### Pasul 1 — Încărcare fișier .npy
Funcție: `load_npy_image(path)`
- Încarcă fișierul NumPy cu `allow_pickle=True`
- Dacă fișierul conține un dicționar (format dataset), extrage cheia `'image'`
- Dacă imaginea e 3D (H×W×3), extrage primul canal
- Asigură că rezultatul e 2D (H×W), dtype float32
- Dimensiune originală imagini dataset: 768×1024 px

### Pasul 2 — Preprocesare
Funcție: `preprocess_for_inference(image)`

**2a. Eliminare text suprapus — `clean_text(image)`:**
1. Conversie la uint8 (`to_uint8`)
2. Filtrare medie locală cu radius=10 folosind summed area table (`mean_filter`)
3. Detecție pixeli text: deviație față de media locală > 22 (în ambele direcții)
4. Eliminare obiecte mici: eroziune morfologică (radius=2) + dilatare (radius=2) — `remove_small_objects`
5. Dilatare mască text cu radius=3
6. Înlocuire pixeli text cu background local (mean_filter radius=12)

**2b. CLAHE:**
- `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`
- Aplicat pe imaginea curățată de text

### Pasul 3 — Transformare tensor
`val_transform = A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0) + ToTensorV2()`
- Normalizare la intervalul [-1, 1]
- Conversie la tensor PyTorch

### Pasul 4 — Inferență model
Funcție: `run_inference(model, image_2d, device)`
- `unsqueeze(0)` → batch size 1
- `torch.no_grad()` pentru eficiență
- `torch.sigmoid(logits) > 0.5` → măști binare
- Output: array NumPy shape (4, H, W), valori 0 sau 1

### Pasul 5 — Afișare rezultate
- Panoul drept: 4 măști în grilă 2×2 (matplotlib embedded)
- Panoul stâng (dacă overlay activ): imagine originală + măști RGBA suprapuse cu opacitate controlabilă

---

## Încărcare model

Funcție: `load_model(device)`
- Model inițializat cu `weights=None` (fără pretrained, se încarcă din fișier)
- `torch.load(MODEL_PATH, map_location=device)`
- Suportă două formate:
  - state_dict direct
  - dicționar checkpoint cu cheia `'model_state_dict'`
- Setat în `eval()` după încărcare
- Device: CUDA dacă disponibil, altfel CPU

---

## Threading

Inferența rulează pe un thread daemon separat pentru a nu bloca UI-ul.
Pe durata inferenței:
- Butoanele Browse și Segment sunt dezactivate
- Loading bar indeterminate este afișat în panoul drept
- Rezultatele sunt trimise înapoi pe thread-ul principal cu `self.after(0, callback)`

---

## Gestionare erori

- Model lipsă (`best_model.pt` negăsit): messagebox.showerror la pornire
- Eroare încărcare imagine: messagebox.showerror cu detalii
- Eroare inferență: messagebox.showerror + UI resetat la starea anterioară

---

## Observații pentru capitol

1. Aplicația implementează **același pipeline de preprocesare** ca Experimentul 5 (eliminare text + CLAHE), deci rezultatele vizuale corespund configurației finale alese.
2. Inferența se face pe **imaginea preprocesată**, nu pe cea brută — utilizatorul vede imaginea originală brută în panoul stâng, dar modelul primește versiunea procesată.
3. Opacitatea implicită a overlay-ului este **70%**, permițând vizualizarea simultană a structurilor anatomice și a predicțiilor.
4. Aplicația nu salvează rezultatele pe disc — este strict un tool de vizualizare interactivă.
5. Aplicația rulează local pe Windows (Tkinter nativ), necesită Python + dependențe instalate.
6. Nu există funcționalitate batch — se procesează câte o imagine la un moment dat.
