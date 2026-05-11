# Introducere

## 1.1 Descrierea temei

Lucrarea de față abordează problema segmentării automate a patru structuri anatomice fetale
— vena ombilicală, artera ombilicală, stomacul și ficatul — în imagini ecografice abdominale
obținute în cadrul examinărilor prenatale de rutină. Segmentarea unei imagini medicale
reprezintă procesul de delimitare a unor regiuni de interes la nivel de pixel, permițând
identificarea și localizarea precisă a structurilor anatomice.

Soluția propusă se bazează pe o arhitectură de rețea neurală convoluțională de tip U-Net,
cu encoder ResNet50 pre-antrenat, adaptat pentru procesarea imaginilor în tonuri de gri
(single-channel). Rețeaua este antrenată să producă simultan patru hărți de segmentare —
câte una pentru fiecare structură —, tratând problema ca o sarcină de segmentare multi-label.

Lucrarea documentează un proces experimental incremental în care, pornind de la un model
de referință (baseline), sunt introduse succesiv îmbunătățiri la nivelul preprocesării
imaginilor și al funcției de antrenare, cu scopul de a evalua contribuția fiecărei modificări
la calitatea segmentării.

---

## 1.2 Motivație

Ecografia abdominală fetală reprezintă investigația de elecție pentru monitorizarea
dezvoltării fătului în perioada prenatală. Examinarea este neinvazivă, lipsită de radiații
ionizante și accesibilă, ceea ce o face standardul de aur pentru screeningul morfologic
efectuat în trimestrul al doilea de sarcină.

Interpretarea imaginilor ecografice este însă o sarcină complexă, dependentă în mare
măsură de experiența operatorului. Calitatea imaginii variază în funcție de aparatură,
poziția fătului și caracteristicile fizice ale pacientei, iar structurile de interes pot
fi dificil de delimitat vizual din cauza contrastului scăzut și a zgomotului specific
acestui tip de imagistică.

O provocare particulară a acestei lucrări o constituie dezechilibrul sever de scară între
structurile segmentate: ficatul fetal poate ocupa o proporție semnificativă din imaginea
ecografică, în timp ce artera ombilicală, o structură tubulară de câțiva milimetri,
ocupă sub 1% din pixelii imaginii. Această diferență face ca metodele clasice de antrenare
să ignore complet structurile mici în favoarea celor mari.

Automatizarea segmentării poate reduce variabilitatea inter-observator, poate accelera
fluxul de lucru în cabinet și poate oferi medicului un instrument de suport pentru
identificarea structurilor anatomice, în special în cazuri dificile.

---

## 1.3 Obiective

Lucrarea urmărește atingerea următoarelor obiective:

1. **Proiectarea unei arhitecturi** capabile să segmenteze simultan patru structuri anatomice
   cu scări și proprietăți vizuale diferite, pornind de la arhitectura U-Net cu encoder
   ResNet50.

2. **Adaptarea unui model pre-antrenat pe date RGB** (ImageNet) la imagini ecografice
   grayscale, prin modificarea primului strat convoluțional pentru intrare single-channel.

3. **Proiectarea unei funcții de loss** care să compenseze dezechilibrul sever dintre
   structurile mici (venă, arteră) și cele mari (stomac, ficat), combinând Binary
   Cross-Entropy cu ponderare per organ și Focal Tversky Loss.

4. **Evaluarea progresului prin experimente incrementale**, fiecare introducând o modificare
   față de modelul anterior:
   
   - Experiment 1: imagini brute (baseline);
   - Experiment 2: eliminarea textului suprapus din imaginile ecografice;
   - Experiment 3: introducerea ponderilor per organ (pos_weight);
   - Experiment 4: aplicarea filtrului CLAHE pentru îmbunătățirea contrastului local;
   - Experiment 5: decupare pătrat a imaginilor combinată cu CLAHE.

5. **Analiza cantitativă și calitativă** a rezultatelor, folosind metricile Dice, IoU,
   Recall (sensibilitate) și Precizie, evaluate separat pentru fiecare organ și pentru
   fiecare configurație experimentală.

---

## 1.4 Structura lucrării

*Această secțiune se completează după finalizarea celorlalte capitole.*
*(Exemplu de formă finală:)*

**Capitolul 1** prezintă contextul medical al lucrării: rolul ecografiei fetale în
screeningul prenatal, anatomia structurilor de interes și provocările specifice
imaginilor ecografice pentru algoritmii de învățare automată.

**Capitolul 2** trece în revistă stadiul actual al cercetării în domeniul segmentării
medicale automate, de la metodele clasice bazate pe procesare de imagine până la
arhitecturile moderne bazate pe rețele neuronale convoluționale, cu accent pe lucrările
relevante pentru segmentarea pe imagini ecografice și pe structuri fetale.

**Capitolul 3** descrie fundamentele teoretice ale metodelor utilizate: rețele
convoluționale, arhitectura ResNet50 și conexiunile reziduale, paradigma U-Net,
transfer learning, funcțiile de loss pentru segmentare și metricile de evaluare.

**Capitolul 4** descrie setul de date, formatul datelor, strategia de împărțire
train/val/test la nivel de pacient și pipeline-ul de augmentare implementat cu
biblioteca Albumentations.

**Capitolul 5** detaliază arhitectura propusă (MultiOutputUNet): encoderul ResNet50
modificat, blocurile decoder cu skip connections, stratul final multi-output și
funcția de loss combinată.

**Capitolul 6** descrie procesul de antrenare: hiperparametrii, optimizatorul,
scheduler-ul, tehnicile de stabilizare (mixed precision, gradient clipping) și
strategia de salvare a checkpoint-urilor.

**Capitolul 7** prezintă cele cinci experimente, rezultatele numerice comparate
și analiza vizuală a predicțiilor.

**Concluziile** sintetizează contribuțiile personale, rezultatele principale și
direcțiile de dezvoltare ulterioară.
