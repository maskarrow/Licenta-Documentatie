# Structura lucrării de licență

**Titlu propus:** Segmentarea multi-organ (venă, arteră, stomac, ficat) pe ecografii abdominale fetale folosind o arhitectură U-Net cu encoder ResNet50

**Autor:** Vlase Andrei-Sebastian, grupa 442A
**Facultatea:** Electronică, Telecomunicații și Tehnologia Informației, UPB

---

## Cuprins

1. **Introducere**
   1.1 Descrierea temei
   1.2 Motivație
   1.3 Obiective
   1.4 Structura lucrării

2. **Capitolul 1 — Context medical și fundamente teoretice**
   2.1 Ecografia abdominală fetală
   2.2 Anatomia structurilor de interes (venă, arteră, stomac, ficat)
   2.3 Provocări specifice imaginilor ecografice
   2.4 Necesitatea segmentării automate

3. **Capitolul 2 — Stadiul actual al cercetării (State of the Art)**
   3.1 Segmentarea în imagistica medicală — abordări clasice și moderne
   3.2 Arhitecturi de tip encoder-decoder pentru segmentare
   3.3 Segmentarea pe imagini ecografice
   3.4 Segmentarea structurilor fetale — particularități și lucrări existente

4. **Capitolul 3 — Fundamente teoretice ale rețelelor neuronale folosite**
   4.1 Rețele neuronale convoluționale (CNN)
   4.2 Arhitectura ResNet50 și conceptul de conexiuni reziduale
   4.3 Arhitectura U-Net și segmentarea semantică
   4.4 Transfer learning și inițializarea cu greutăți pre-antrenate
   4.5 Funcții de loss pentru segmentare (BCE, Dice, Tversky, Focal Tversky)
   4.6 Metrici de evaluare (Dice, IoU, sensibilitate, specificitate, precizie)

5. **Capitolul 4 — Setul de date și preprocesarea**
   4.1 Descrierea setului de date
   4.2 Formatul datelor și organizarea fișierelor
   4.3 Împărțirea train/val/test la nivel de pacient
   4.4 Augmentarea datelor (Albumentations)
   4.5 Adaptarea single-channel pentru intrarea ResNet50

6. **Capitolul 5 — Arhitectura propusă (MultiOutputUNet)**
   5.1 Schema generală a arhitecturii
   5.2 Encoderul bazat pe ResNet50 modificat
   5.3 Blocurile decoder cu skip connections
   5.4 Stratul final și ieșirile multi-organ
   5.5 Funcția de loss combinată (BCE ponderat + Focal Tversky)

7. **Capitolul 6 — Procesul de antrenare**
   6.1 Configurația experimentală (hiperparametri, optimizator, scheduler)
   6.2 Mixed Precision Training (AMP) și gradient clipping
   6.3 Strategia de salvare a checkpoint-urilor
   6.4 Mediul de antrenare (hardware, software)

8. **Capitolul 7 — Experimente și rezultate**
   7.1 Experimentul 1 — Antrenare pe imagini brute (baseline)
   7.2 Experimentul 2 — Antrenare pe imagini fără text suprapus
   7.3 Experimentul 3 — Antrenare cu ponderi (pos_weights) ajustate per organ
   7.4 Experimentul 4 — Antrenare cu filtru CLAHE
   7.5 Experimentul 5 — Antrenare cu imagini decupate pătrat (+ CLAHE)
   7.6 Comparație globală între experimente
   7.7 Analiză vizuală a predicțiilor

9. **Concluzii și direcții viitoare**
   8.1 Concluzii
   8.2 Contribuții personale
   8.3 Direcții viitoare

10. **Bibliografie**

11. **Anexe** (cod sursă relevant, tabele detaliate, figuri suplimentare)

---

# Introducere

## 1.1 Descrierea temei

Aici trebuie să explici pe scurt (3-5 propoziții) ce înseamnă segmentarea multi-organ pe ecografii fetale abdominale. Menționează cele 4 structuri vizate (venă, arteră, stomac, ficat) și de ce sunt importante medical. Descrie pe scurt soluția propusă: o rețea de tip U-Net cu encoder ResNet50 adaptat la intrare single-channel. Spune că lucrarea explorează mai multe variante de preprocesare și funcții de loss pentru a îmbunătăți progresiv performanța.

## 1.2 Motivație

Aici trebuie să explici de ce este utilă această lucrare. Argumentează că ecografia fetală este o investigație standard, dar interpretarea ei este consumatoare de timp și depinde de experiența medicului. Explică de ce segmentarea automată ajută (măsurători obiective, screening rapid, reducerea variabilității inter-observator). Adaugă motivația tehnică: structurile diferă enorm ca dimensiune (artera și vena ombilicală sunt foarte mici, ficatul și stomacul sunt mari), ceea ce face problema interesantă din punct de vedere al dezbalansului de clase.

## 1.3 Obiective

Aici trebuie să listezi clar obiectivele lucrării (sub formă de listă numerotată sau bullet points). Exemple de obiective:

- proiectarea unei arhitecturi capabile să segmenteze simultan 4 structuri anatomice diferite ca scară;
- adaptarea unui encoder pre-antrenat pe ImageNet (ResNet50, 3 canale) la intrare grayscale;
- proiectarea unei funcții de loss care să compenseze dezbalansul sever între organe mici și mari;
- evaluarea progresului prin mai multe experimente succesive, fiecare adăugând o îmbunătățire (curățare text, ponderi, filtre CLAHE, crop pătrat);
- analiza cantitativă (Dice, IoU) și calitativă (predicții vizuale) a rezultatelor.

## 1.4 Structura lucrării

Aici trebuie să descrii pe scurt fiecare capitol (1-2 propoziții per capitol). Scrii această secțiune ULTIMA, după ce restul lucrării e gata, ca să fie acurată.

---

# Capitolul 1 — Context medical și fundamente teoretice

## 1.1 Ecografia abdominală fetală

Aici trebuie să explici ce este ecografia și de ce este standardul de aur pentru investigația prenatală. Menționează că folosește ultrasunete (non-invazivă, fără radiații), că imaginile sunt 2D (sau 3D/4D, dar tu lucrezi pe 2D), și că calitatea depinde de operator, sondă și poziția fătului. Explică pe scurt ce este un examen de morfologie fetală și momentul în care se face (de obicei trimestrul II).

## 1.2 Anatomia structurilor de interes

Aici trebuie să prezinți cele 4 structuri, una câte una. Pentru fiecare, descrie:

- ce este (vena ombilicală, artera ombilicală, stomac fetal, ficat fetal);
- dimensiunea relativă pe ecografie (artera și vena — structuri tubulare mici; stomac — bulă anecoică medie; ficat — masă mare omogenă);
- de ce este importantă vizualizarea ei la screening (de exemplu artera ombilicală unică, stomac neidentificabil → suspiciune atrezie esofagiană, etc.).

Recomandare: include o figură anatomică cu o ecografie reală etichetată din setul tău de date.

## 1.3 Provocări specifice imaginilor ecografice

Aici trebuie să explici de ce ecografia este grea pentru deep learning. Listează:

- speckle noise (zgomot multiplicativ caracteristic);
- contrast scăzut între structuri;
- artefacte de umbrire acustică;
- variabilitate mare între pacienți și operatori;
- text suprapus și margini negre (artefacte de la aparat) — exact ce ai eliminat tu în experimentul 2;
- dezbalansul ariei structurilor (artera ocupă <0.5% din imagine, ficatul poate ocupa peste 30%).

## 1.4 Necesitatea segmentării automate

Aici trebuie să argumentezi de ce o soluție automată este utilă în practică. Menționează că poate ajuta medicii ca instrument de suport (NU înlocuire), că poate accelera screening-ul și poate oferi măsurători obiective reproductibile. Subliniază că lucrarea ta este o etapă de cercetare aplicată, nu un produs medical certificat.

---

# Capitolul 2 — Stadiul actual al cercetării (State of the Art)

## 2.1 Segmentarea în imagistica medicală — abordări clasice și moderne

Aici trebuie să faci o trecere în revistă scurtă a evoluției: metode clasice (thresholding, region growing, level sets, active contours) → metode bazate pe CNN (FCN-2015, U-Net-2015) → arhitecturi moderne (Attention U-Net, TransUNet, nnU-Net, Segment Anything Model). Pentru fiecare familie de metode, menționează 1-2 lucrări reprezentative cu citare în bibliografie. Argumentează de ce ai ales abordarea CNN pe baza U-Net.

## 2.2 Arhitecturi de tip encoder-decoder pentru segmentare

Aici trebuie să explici de ce paradigma encoder-decoder este dominantă în segmentarea medicală. Prezintă U-Net original (Ronneberger et al., 2015) ca punct de pornire. Explică ce înseamnă „backbone” (encoder) și de ce se înlocuiește frecvent cu rețele puternice pre-antrenate (VGG, ResNet, EfficientNet). Justifică de ce ai ales ResNet50 ca encoder (echilibru între capacitate și cost computațional, popularitate, greutăți ImageNet disponibile).

## 2.3 Segmentarea pe imagini ecografice

Aici trebuie să prezinți lucrări specifice de segmentare pe ultrasunete. Caută și citează 3-5 lucrări recente (2020-2025) pe topici similare: segmentare carotidă, tiroidă, mușchi, organe abdominale adulte pe ecografie. Subliniază că majoritatea lucrărilor pe ecografie folosesc U-Net sau variante. Evidențiază că pe ecografii canalul de intrare este natural unul singur (grayscale), motiv pentru care adaptarea ResNet50 (proiectat pentru 3 canale RGB) este necesară.

## 2.4 Segmentarea structurilor fetale

Aici trebuie să prezinți lucrările cele mai apropiate de tema ta: segmentarea structurilor fetale (creier, inimă, abdomen) pe ecografie. Caută lucrări din challenge-uri (de exemplu HC18 — head circumference, sau diverse provocări MICCAI). Subliniază contribuția ta personală: după cunoștința ta nu există un set de lucrări consacrate care să facă simultan segmentare pe venă + arteră + stomac + ficat la făt — aici ai un unghi de noutate. Verifică acest lucru și formulează corect (nu spune că nu există, ci „lucrările existente segmentează izolat structuri individuale”).

---

# Capitolul 3 — Fundamente teoretice ale rețelelor neuronale folosite

## 3.1 Rețele neuronale convoluționale (CNN)

Aici trebuie să explici ce este un CNN: noțiunea de convoluție 2D, kernel, stride, padding, feature map. Adaugă o figură schematică simplă. Menționează rolul straturilor de pooling (downsampling) și al funcțiilor de activare (ReLU). Explică intuiția: straturile inferioare detectează muchii și texturi, straturile superioare detectează concepte complexe.

## 3.2 Arhitectura ResNet50 și conexiunile reziduale

Aici trebuie să explici ce este ResNet50 și de ce a fost o inovație (He et al., 2015). Detaliază:

- problema „vanishing gradient” în rețele adânci și soluția prin skip connections;
- structura blocurilor bottleneck (1x1, 3x3, 1x1);
- cele 4 etape principale (layer1-layer4) cu numărul de canale 256, 512, 1024, 2048;
- modul în care folosești această ierarhie de feature maps ca encoder (enc1-enc5 în codul tău).

Include o figură cu schema unui bloc rezidual.

## 3.3 Arhitectura U-Net și segmentarea semantică

Aici trebuie să prezinți U-Net (Ronneberger, 2015). Explică:

- partea de encoder (contracție) — extrage features la rezoluții descrescătoare;
- partea de decoder (expansiune) — reconstruiește harta de segmentare la rezoluția originală;
- skip connections — esențiale pentru a păstra detaliile spațiale fine pierdute la downsampling;
- output-ul — o hartă de probabilități per pixel pentru fiecare clasă.

Justifică de ce U-Net este ideal pentru segmentare medicală: funcționează bine cu seturi mici, păstrează contururile fine.

## 3.4 Transfer learning și inițializarea cu greutăți pre-antrenate

Aici trebuie să explici conceptul de transfer learning: în loc să antrenezi de la zero, pornești de la o rețea antrenată pe un set mare (ImageNet) și o adaptezi la sarcina ta. Explică PRECIS soluția ta tehnică pentru a folosi greutățile RGB pe imagini single-channel: ai sumat cele 3 canale ale primului strat convoluțional (`torch.sum(weights, dim=1, keepdim=True)`). Argumentează de ce această soluție este preferată în locul mediei (păstrează magnitudinea răspunsului) sau a inițializării random (pierde transferul de cunoștințe).

## 3.5 Funcții de loss pentru segmentare

Aici trebuie să prezinți și să justifici funcțiile de loss. Pentru fiecare, dă formula matematică și intuiția:

- **Binary Cross-Entropy (BCE)** — penalizează pixel cu pixel; problemă: ignoră dezbalansul.
- **BCE cu pos_weight** — multiplici contribuția pixelilor pozitivi (organ) cu un factor; soluția ta pentru artere foarte mici.
- **Dice Loss** — măsoară overlap-ul direct; bună pentru clase rare.
- **Tversky Loss** — generalizare a Dice cu parametri α și β pentru a controla balanța FP vs FN.
- **Focal Tversky** — adaugă exponentul γ pentru a se concentra pe exemplele greu de clasificat.

Justifică combinația ta finală: `0.4 * BCE_weighted + 0.6 * FocalTversky` cu α=0.4, β=0.6, γ=0.75. Explică ce face fiecare parametru.

## 3.6 Metrici de evaluare

Aici trebuie să prezinți metricile pe care le folosești și ce înseamnă fiecare:

- **Dice (F1)** — `2*TP / (2*TP + FP + FN)`, standardul în segmentare medicală;
- **IoU (Jaccard)** — `TP / (TP + FP + FN)`, mai conservatoare ca Dice;
- **Sensibilitate (Recall)** — `TP / (TP + FN)`, cât din organul real ai prins;
- **Specificitate** — `TN / (TN + FP)`, cât din fundal ai identificat corect;
- **Precizie** — `TP / (TP + FP)`, cât din predicția ta e corectă.

Explică de ce raportezi mai multe metrici și nu doar Dice (sensibilitatea și specificitatea ajută la diagnosticarea problemelor — de exemplu Dice mic + specificitate mare = under-segmentation).

---

# Capitolul 4 — Setul de date și preprocesarea

## 4.1 Descrierea setului de date

Aici trebuie să descrii setul tău: câte imagini, câți pacienți, cum au fost obținute (consultă-te cu coordonatorul pentru detalii medicale — etică, anonimizare, sursă), rezoluția originală, formatele în care vin etichetele. Adaugă un tabel cu statistici (număr imagini per organ, procent mediu de pixeli ocupat de fiecare organ).

## 4.2 Formatul datelor și organizarea fișierelor

Aici trebuie să explici formatul `.npy` în care îți vin datele. Codul tău încarcă un dicționar cu `image` (HxWx1) și `structures` (dict de măști per organ alfabetic). Explică de ce s-a ales acest format (rapid de citit, păstrează tot într-un fișier). Include exemplul de cod:

```python
data = np.load(self.paths[idx], allow_pickle=True).item()
image = data['image'][..., 0:1].astype(np.float32)
masks_list = [data['structures'][organ].astype(np.float32) for organ in self.organ_names]
```

## 4.3 Împărțirea train/val/test la nivel de pacient

Aici trebuie să explici de ce împărțirea trebuie făcută la nivel de PACIENT, nu de imagine. Dacă două imagini de la același pacient ajung una în train și una în test, ai data leakage și performanța de test va fi nerealist de bună. Codul tău rezolvă asta extragând ID-ul pacientului (`P\d+`) din numele fișierului și împărțind apoi cu `train_test_split` pacienții. Raportul tău: 70% train / 15% val / 15% test. Include code snippet:

```python
patients = sorted(set(re.search(r'(P\d+)_IMG', os.path.basename(f)).group(1) for f in all_files))
train_p, temp_p = train_test_split(patients, test_size=val_ratio + test_ratio, random_state=seed)
val_p, test_p = train_test_split(temp_p, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)
```

## 4.4 Augmentarea datelor (Albumentations)

Aici trebuie să explici de ce augmentarea este esențială cu seturi medicale mici. Prezintă fiecare transformare din pipeline-ul tău și de ce ai ales-o:

- `Resize(256, 256)` — uniformizează intrarea pentru rețea;
- `HorizontalFlip` — anatomic plauzibil pe ecografii fetale;
- `RandomRotate90` — explică DACĂ este plauzibilă (rotații mari schimbă orientarea anatomică — poate vrei să o discuți critic);
- `RandomBrightnessContrast` — simulează variații între aparate;
- `GaussNoise` — simulează speckle ecografic;
- `ElasticTransform` — simulează deformări anatomice;
- `Normalize(mean=0, std=1)` — păstrează intervalul.

Subliniază că Albumentations aplică SIMULTAN aceeași transformare imaginii și tuturor măștilor — esențial pentru segmentare.

## 4.5 Adaptarea single-channel

Aici trebuie să explici concret modificarea pe care ai făcut-o la primul strat ResNet50. Include code snippet și explicația matematică:

```python
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
weights = resnet.conv1.weight.clone()                    # (64, 3, 7, 7)
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet.conv1.weight.data = torch.sum(weights, dim=1, keepdim=True)  # (64, 1, 7, 7)
```

Argumentează de ce ai ales suma celor 3 canale RGB peste alternative (medie, doar canalul verde, inițializare aleatoare).

---

# Capitolul 5 — Arhitectura propusă (MultiOutputUNet)

## 5.1 Schema generală

Aici trebuie să prezinți o figură-schemă cu întreaga arhitectură: input grayscale 1x256x256 → encoder ResNet50 (5 niveluri) → decoder cu 4 blocuri + upsample final → output multi-canal Cx256x256 unde C = numărul de organe. Explică intuiția: encoderul comprimă informația, decoderul o reconstruiește la rezoluția originală, skip connections aduc detaliile spațiale.

## 5.2 Encoderul bazat pe ResNet50

Aici trebuie să detaliezi compoziția encoderului cu dimensiunile pe fiecare nivel. Folosește un tabel:

| Nivel | Modul          | Output channels | Rezoluție (input 256x256) |
| ----- | -------------- | --------------- | ------------------------- |
| enc1  | conv1+bn+relu  | 64              | 128x128                   |
| enc2  | maxpool+layer1 | 256             | 64x64                     |
| enc3  | layer2         | 512             | 32x32                     |
| enc4  | layer3         | 1024            | 16x16                     |
| enc5  | layer4         | 2048            | 8x8                       |

Explică ce face fiecare nivel.

## 5.3 Blocurile decoder cu skip connections

Aici trebuie să prezinți structura unui `DecoderBlock` din codul tău. Include code snippet:

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
```

Explică:

- `Upsample bilinear` în loc de `ConvTranspose2d` — evită checkerboard artifacts și nu adaugă parametri antrenabili extra;
- concatenarea cu skip-ul de la encoder pe dimensiunea canalelor;
- două convoluții 3x3 cu BN și ReLU.

Adaugă un tabel cu cele 4 blocuri decoder și dimensiunile lor (dec4, dec3, dec2, dec1).

## 5.4 Stratul final și ieșirile multi-organ

Aici trebuie să explici ce face `final_conv` din codul tău: un upsample x2 (de la 128x128 la 256x256) urmat de o convoluție 1x1 care produce `num_classes` canale de output. Subliniază că folosești **sigmoid pe fiecare canal** (nu softmax) pentru că organele pot suprapune (vena, artera, stomac, ficat pot apărea simultan în aceeași imagine și un pixel poate aparține teoretic mai multor structuri). Aceasta este o decizie de design importantă: tratezi problema ca segmentare multi-label, nu multi-class.

## 5.5 Funcția de loss combinată

Aici trebuie să explici detaliat clasa `MultiOrganLoss`:

```python
pos_weights = torch.tensor([150.0, 15.0, 20.0, 30.0]).view(1, -1, 1, 1)
self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
```

Argumentează valorile alese pentru ['artery', 'liver', 'stomach', 'vein']:

- artera (150) — cea mai mică, are nevoie de cel mai mare weight;
- ficatul (15) — cel mai mare, weight mic;
- stomacul (20) — mediu;
- vena (30) — mică, dar nu la fel de greu de detectat ca artera.

Apoi componenta Focal Tversky: explică formulele cu α=0.4, β=0.6, γ=0.75 și de ce această alegere penalizează mai mult FN decât FP (preferi să prinzi tot organul chiar cu risc de over-segmentation). Combinația finală: `0.4 * BCE + 0.6 * FocalTversky`.

---

# Capitolul 6 — Procesul de antrenare

## 6.1 Configurația experimentală

Aici trebuie să prezinți într-un tabel hiperparametrii folosiți:

| Parametru             | Valoare                                       |
| --------------------- | --------------------------------------------- |
| Batch size            | 8                                             |
| Learning rate inițial | 3e-4                                          |
| Optimizator           | AdamW                                         |
| Weight decay          | 1e-5                                          |
| Scheduler             | ReduceLROnPlateau                             |
| Epoch-uri totale      | 30/60/90/120/150 (variabil între experimente) |
| Max grad norm         | 1.0                                           |
| Seed                  | 42                                            |

Argumentează fiecare alegere (de ce AdamW și nu SGD, de ce learning rate-ul ăsta, de ce batch 8).

## 6.2 Mixed Precision Training (AMP) și gradient clipping

Aici trebuie să explici de ce ai folosit `autocast` + `GradScaler`: antrenare mai rapidă pe GPU prin folosirea fp16 pentru operații, păstrând fp32 pentru stabilitate numerică. Reduce VRAM și accelerează antrenamentul. Explică și gradient clipping (`clip_grad_norm_` cu max_norm=1.0): previne explozia gradienților, care apare uneori la combinarea BCE cu pos_weight mare și loss-uri compuse.

## 6.3 Strategia de salvare a checkpoint-urilor

Aici trebuie să explici strategia ta: salvezi modele la fiecare 30 de epoci (30, 60, 90, 120, 150). Justifică: îți permite să compari progresia învățării în timp și să faci un grafic „Dice vs epoci”. Menționează că salvezi `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, și `best_val_dice` pentru a putea relua antrenarea.

## 6.4 Mediul de antrenare

Aici trebuie să detaliezi: GPU folosit (de ex. Tesla T4 / V100 / RTX local), CUDA version, PyTorch version (din `torch.__version__`), Albumentations version. Menționează dacă ai folosit Google Colab (Pro?), Kaggle sau setup local. Include și timpul mediu per epocă.

---

# Capitolul 7 — Experimente și rezultate

**Notă structurală:** fiecare experiment trebuie prezentat în același tipar pentru comparabilitate: (a) motivația modificării, (b) descrierea concretă, (c) tabel cu rezultate per organ la fiecare punct de checkpoint, (d) discuție, (e) figură cu predicții.

## 7.1 Experimentul 1 — Baseline (imagini brute)

Aici trebuie să descrii primul experiment ca punct de referință. Spune că ai antrenat pe imaginile brute cu text suprapus și margini negre (artefactele aparatului), fără ponderi diferențiate (pos_weight=1 pentru toate). Prezintă rezultatele la 30, 60, 90, 120 epoci într-un tabel cu Dice/IoU per organ. Discută ce ai observat: probabil ficatul și stomacul ies bine, vena și artera ies prost — exact problema pe care vrei să o rezolvi.

## 7.2 Experimentul 2 — Imagini fără text suprapus

Aici trebuie să explici motivația: textul de pe ecografii (dată, parametri sondă, etichete medicale) este zgomot care nu are relevanță anatomică și consumă din capacitatea rețelei. Descrie cum ai eliminat textul (procesare offline cu mască peste regiunile cu text). Prezintă rezultatele și compară-le cu baseline-ul. Așteaptă-te la o îmbunătățire moderată — discut-o.

## 7.3 Experimentul 3 — Ponderi pos_weight ajustate per organ

Aici trebuie să descrii adăugarea pos_weights [150, 15, 20, 30] pentru artery/liver/stomach/vein. Argumentează valorile prin raportul aproximativ de pixeli (ar trebui să fie invers proporțional cu frecvența pixelilor pozitivi). Prezintă rezultatele și subliniază îmbunătățirea pe organele mici (artera și vena). Aceasta este probabil **cea mai importantă îmbunătățire** din lucrarea ta — tratează-o pe larg.

## 7.4 Experimentul 4 — CLAHE

Aici trebuie să explici ce este CLAHE (Contrast Limited Adaptive Histogram Equalization): metodă de îmbunătățire locală a contrastului care nu amplifică zgomotul ca histograma echalization clasică. Argumentează de ce este potrivită pentru ecografii (contrast slab, structuri tubulare mici greu de văzut). Descrie parametrii folosiți (clip_limit, tile_grid_size). Prezintă rezultatele.

## 7.5 Experimentul 5 — Crop pătrat + CLAHE

Aici trebuie să motivezi crop-ul pătrat: imaginile ecografice originale au margini negre laterale care reprezintă conul de scanare. Decuparea pătrată centrată elimină pixelii inutili și permite o redimensionare la 256x256 fără distorsiune anizotropă. Combinat cu CLAHE, ar trebui să dea cel mai bun rezultat. Prezintă rezultatele.

## 7.6 Comparație globală

Aici trebuie să incluzi:

- un tabel mare cu toate experimentele și toate organele la cea mai bună epocă a fiecăruia;
- un grafic „Dice mediu vs epoci” cu o curbă per experiment;
- un grafic „Dice per organ” bar chart cu grupuri pe experiment;
- discuție care e cea mai bună combinație și de ce.

## 7.7 Analiză vizuală a predicțiilor

Aici trebuie să prezinți cazuri vizuale: 3-5 exemple unde modelul funcționează bine și 2-3 cazuri de eșec (failure cases). Pentru fiecare, arată imaginea originală, masca ground truth și predicția modelului, pe 3 rânduri x 4 coloane (un organ per coloană). Folosește funcția `plot_results` din notebook-ul tău. Comentează critic: unde greșește modelul, ce ar putea îmbunătăți rezultatul.

---

# Concluzii și direcții viitoare

## 8.1 Concluzii

Aici trebuie să rezumi rezultatele principale în 5-7 propoziții. Spune că ai construit o arhitectură funcțională, că ai demonstrat un progres clar prin experimentele incrementale, că cele mai mari câștiguri au venit din [completezi după rezultate — probabil ponderile per organ + CLAHE]. Subliniază că modelul atinge performanțe utile pentru organele mari, mai modeste pentru organele mici, ceea ce este consistent cu literatura.

## 8.2 Contribuții personale

Aici trebuie să listezi clar ce ai făcut TU (nu literatura). Listă:

- adaptarea ResNet50 ImageNet la intrare single-channel prin sumarea canalelor;
- proiectarea funcției de loss combinate cu pos_weights specifici fiecărui organ;
- realizarea pipeline-ului complet de antrenare cu AMP, gradient clipping, ReduceLROnPlateau;
- proiectarea celor 5 experimente incrementale și analiza comparativă;
- (dacă e cazul) preprocesarea pentru eliminarea textului și crop pătrat.

Această secțiune este CRUCIALĂ la comisie. Pregătește-o cu grijă.

## 8.3 Direcții viitoare

Aici trebuie să sugerezi cum poate fi extinsă lucrarea. Idei:

- antrenare pe set mai mare cu mai mulți pacienți;
- comparație cu arhitecturi mai noi (Attention U-Net, TransUNet, SAM-Medical);
- adăugarea unei componente de post-procesare (CRF, morphology);
- evaluare cu metrici clinice (de ex. acordul cu măsurătorile manuale ale medicilor);
- extindere la 3D dacă există volume.

---

# Bibliografie

Listează aici sursele citate, numerotate. Reguli din template (format românesc):

```
[1] Ronneberger, O., Fischer, P., Brox, T., "U-Net: Convolutional Networks for Biomedical Image Segmentation",
    în MICCAI 2015, pp. 234-241.

[2] He, K., Zhang, X., Ren, S., Sun, J., "Deep Residual Learning for Image Recognition",
    în CVPR 2016, pp. 770-778.

[3] Salehi, S.S.M., Erdogmus, D., Gholipour, A., "Tversky Loss Function for Image Segmentation Using 3D
    Fully Convolutional Deep Networks", în MLMI 2017.

[4] Buslaev, A., et al., "Albumentations: Fast and Flexible Image Augmentations",
    în Information, vol. 11, nr. 2/2020, pp. 125.

[5] Paszke, A., et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library",
    în NeurIPS 2019.
```

Minim 15-20 surse pentru o licență decentă. Caută activ pe Google Scholar lucrări recente pe „fetal ultrasound segmentation”, „multi-organ segmentation U-Net”, „ResNet encoder segmentation”.

---

# Anexe

## Anexa 1 — Cod sursă reprezentativ

Include cu font monospace (Courier New 10pt, conform regulamentului UPB):

- definiția completă `MultiOutputUNet`;
- definiția completă `MultiOrganLoss`;
- funcția `compute_metrics`;
- funcția `analyze_and_split_dataset`;
- bucla principală de antrenare.

## Anexa 2 — Tabele detaliate cu rezultate

Pentru fiecare experiment, tabel complet: epoch x organ x metric (5 metrici).

---

# Sfaturi finale

1. **Scrie întâi schiletul în propoziții scurte**, apoi extinde fiecare capitol. Nu încerca să scrii cap-coadă din prima.
2. **Începe cu capitolele 4-7** (concrete, ai datele) și lasă introducerea + state of the art la final, când ai imaginea completă.
3. **Bibliografia se adaugă PE PARCURS**, niciodată la final — vei uita ce ai citit.
4. **Toate figurile** trebuie să aibă caption, număr și să fie referite în text (de ex. „așa cum se vede în Figura 5.1, ...”).
5. **Toate tabelele și ecuațiile** la fel — numerotare automată în Word (vezi note din template).
6. **Diacritice** de la început, nu la final.
7. **Pentru fiecare experiment**, salvează ACUM curbele de loss/dice — vor fi necesare în capitolul 7 și e mai greu să le regenerezi mai târziu.
