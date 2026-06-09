import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch
import matplotlib.pyplot as plt


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# culori interfata
BG_MAIN     = "#1b1b20"
BG_FRAME    = "#232329"
BG_CARD     = "#2a2a31"
BG_RAISED   = "#33333b"
CLR_BORDER  = "#3a3a44"
CLR_TEXT    = "#e8e8ee"
CLR_SUBTLE  = "#8b8b97"
CLR_DIM     = "#5f5f6b"
CLR_ACCENT  = "#8b5cf6"
CLR_ACCENT2 = "#7c4dfb"
CLR_OK      = "#22c55e"
BG_VIEW     = "#0e0e10"

CLASE    = ['artery', 'liver', 'stomach', 'vein']
NR_CLASE = 4

COLORS = {
    'artery':  "#f59e0b",
    'liver':   "#ef4444",
    'stomach': "#22c55e",
    'vein':    "#3b82f6",
}


def hex_to_float(hex_str):
    h = hex_str.lstrip('#')
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b)


COLORS_RGB = {cls: hex_to_float(col) for cls, col in COLORS.items()}

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_model.pt")

aug_val = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
    ToTensorV2()
])


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


def incarca_model(device):
    net = SegModel(NR_CLASE).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    net.load_state_dict(ckpt)
    net.eval()
    return net


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


def inferenta(model, img2d, device):
    aug = aug_val(image=img2d)
    t = aug['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(t)
        pred = (torch.sigmoid(logits) > 0.5).float()
    return pred[0].cpu().numpy()


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Segmentare Organe")
        self.geometry("1280x820")
        self.minsize(1200, 760)
        self.configure(fg_color=BG_MAIN)

        self._cale_imagine  = None
        self._imagine_2d    = None
        self._pred          = None
        self._model         = None
        self._overlay_activ = False
        self._opacitate     = 0.70
        self._organ_viz     = {n: True for n in CLASE}

        self._fig_st = self._ax_st = self._canvas_st = None
        self._fig_dr = self._axe_dr = self._canvas_dr = None

        self._construieste_ui()
        self._initializeaza_model()

    def _construieste_ui(self):
        self._bara_status()
        self._bara_instrumente()
        self._legenda()
        self._panouri()

    def _bara_instrumente(self):
        tb = ctk.CTkFrame(self, fg_color=BG_FRAME, corner_radius=0, height=62)
        tb.pack(side="top", fill="x")
        tb.pack_propagate(False)
        tb.grid_columnconfigure(7, weight=1)

        font_btn = ctk.CTkFont(size=13, weight="bold")

        self._btn_deschide = ctk.CTkButton(
            tb, text="Browse Image", width=148, height=38,
            corner_radius=9, fg_color=CLR_ACCENT, hover_color=CLR_ACCENT2,
            text_color=CLR_TEXT, font=font_btn, command=self._deschide_fisier
        )
        self._btn_deschide.grid(row=0, column=0, padx=(16, 6), pady=12)

        self._btn_segment = ctk.CTkButton(
            tb, text="Segment", width=120, height=38,
            corner_radius=9, fg_color=CLR_ACCENT, hover_color=CLR_ACCENT2,
            text_color=CLR_TEXT, font=font_btn, state="disabled",
            command=self._ruleaza_segmentare
        )
        self._btn_segment.grid(row=0, column=1, padx=6, pady=12)

        self._btn_overlay = ctk.CTkButton(
            tb, text="Overlay View", width=138, height=38,
            corner_radius=9, fg_color=BG_RAISED, hover_color="#3d3d46",
            text_color=CLR_SUBTLE, font=ctk.CTkFont(size=13),
            state="disabled", command=self._comuta_overlay
        )
        self._btn_overlay.grid(row=0, column=2, padx=6, pady=12)

        ctk.CTkFrame(tb, width=1, height=26, fg_color=CLR_BORDER).grid(
            row=0, column=3, padx=12, pady=18
        )

        ctk.CTkLabel(
            tb, text="Opacitate măști", text_color=CLR_SUBTLE,
            font=ctk.CTkFont(size=12)
        ).grid(row=0, column=4, padx=(0, 4), pady=12)

        self._slider = ctk.CTkSlider(
            tb, from_=0, to=100, number_of_steps=100, width=140,
            progress_color=CLR_ACCENT, button_color=CLR_TEXT,
            button_hover_color=CLR_ACCENT, state="disabled",
            command=self._schimba_opacitate
        )
        self._slider.set(70)
        self._slider.grid(row=0, column=5, padx=4, pady=12)

        self._lbl_opacitate = ctk.CTkLabel(
            tb, text="70%", text_color=CLR_SUBTLE, width=38,
            font=ctk.CTkFont(family="Courier New", size=12)
        )
        self._lbl_opacitate.grid(row=0, column=6, padx=(2, 10), pady=12)

        ctk.CTkFrame(tb, fg_color="transparent").grid(row=0, column=7, sticky="ew")

        pastila = ctk.CTkFrame(tb, fg_color=BG_CARD, corner_radius=8, height=34)
        pastila.grid(row=0, column=8, padx=(0, 16), pady=14)

        self._dot_status = ctk.CTkLabel(
            pastila, text="●", text_color=CLR_DIM,
            font=ctk.CTkFont(size=10), width=18
        )
        self._dot_status.pack(side="left", padx=(8, 2), pady=4)

        self._txt_status = ctk.CTkLabel(
            pastila, text="Niciun fișier selectat.",
            text_color=CLR_SUBTLE, font=ctk.CTkFont(size=12), width=210
        )
        self._txt_status.pack(side="left", padx=(0, 10), pady=4)

    def _legenda(self):
        leg = ctk.CTkFrame(self, fg_color=BG_MAIN, height=48)
        leg.pack(side="top", fill="x", padx=16, pady=(8, 0))
        leg.pack_propagate(False)

        ctk.CTkLabel(
            leg, text="ORGANE", text_color=CLR_DIM,
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(side="left", padx=(0, 14))

        self._chips = {}
        for cls in CLASE:
            chip = self._creeaza_chip(leg, cls)
            chip["frame"].pack(side="left", padx=5)
            self._chips[cls] = chip

    def _creeaza_chip(self, parent, cls):
        culoare = COLORS[cls]
        frame = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=15, height=32)
        frame.pack_propagate(False)

        swatch = ctk.CTkLabel(frame, text="", fg_color=culoare, corner_radius=3, width=14, height=14)
        swatch.pack(side="left", padx=(10, 5), pady=9)

        label = ctk.CTkLabel(
            frame, text=cls.capitalize(),
            text_color=CLR_TEXT, font=ctk.CTkFont(size=12)
        )
        label.pack(side="left", padx=(0, 12), pady=9)

        def _click(_event=None, n=cls):
            self._comuta_organ(n)

        for w in (frame, swatch, label):
            w.bind("<Button-1>", _click)
            w.configure(cursor="hand2")

        return {"frame": frame, "swatch": swatch, "label": label, "active": True}

    def _panouri(self):
        cont = ctk.CTkFrame(self, fg_color=BG_MAIN)
        cont.pack(side="top", fill="both", expand=True, padx=16, pady=10)
        cont.grid_columnconfigure(0, weight=1)
        cont.grid_columnconfigure(1, weight=1)
        cont.grid_rowconfigure(0, weight=1)

        self._pan_st = self._creeaza_panou(cont, "Imagine originală", col=0)
        self._pan_dr = self._creeaza_panou(cont, "Măști segmentare", col=1)

        self._ph_st = tk.Label(
            self._pan_st["vp_tk"], text="Selectează un fișier .npy",
            fg=CLR_DIM, bg=BG_VIEW, font=("Segoe UI", 11)
        )
        self._ph_st.place(relx=0.5, rely=0.5, anchor="center")

        self._ph_dr = tk.Label(
            self._pan_dr["vp_tk"], text="Apasă Segment pentru a vedea măștile",
            fg=CLR_DIM, bg=BG_VIEW, font=("Segoe UI", 11)
        )
        self._ph_dr.place(relx=0.5, rely=0.5, anchor="center")

        self._frame_incarcare = tk.Frame(self._pan_dr["vp_tk"], bg=BG_VIEW)
        self._bara_incarcare = ctk.CTkProgressBar(
            self._frame_incarcare, mode="indeterminate",
            progress_color=CLR_ACCENT, width=200, height=6
        )
        self._bara_incarcare.pack()
        tk.Label(
            self._frame_incarcare, text="Segmentare în curs…",
            fg=CLR_SUBTLE, bg=BG_VIEW, font=("Segoe UI", 11)
        ).pack(pady=(10, 0))

    def _creeaza_panou(self, parent, titlu, col):
        padx = (0, 7) if col == 0 else (7, 0)
        frame = ctk.CTkFrame(
            parent, fg_color=BG_FRAME, corner_radius=12,
            border_width=1, border_color=CLR_BORDER
        )
        frame.grid(row=0, column=col, sticky="nsew", padx=padx)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=0, height=42)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text=titlu, text_color=CLR_TEXT,
            font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
        ).grid(row=0, column=0, padx=14, sticky="w", pady=10)

        lbl_info = ctk.CTkLabel(
            header, text="—", text_color=CLR_DIM,
            font=ctk.CTkFont(family="Courier New", size=11), anchor="e"
        )
        lbl_info.grid(row=0, column=1, padx=14, sticky="e", pady=10)

        vp = ctk.CTkFrame(frame, fg_color=BG_VIEW, corner_radius=0)
        vp.grid(row=1, column=0, sticky="nsew")
        vp.grid_columnconfigure(0, weight=1)
        vp.grid_rowconfigure(0, weight=1)

        vp_tk = tk.Frame(vp, bg=BG_VIEW)
        vp_tk.grid(row=0, column=0, sticky="nsew")

        return {"frame": frame, "header": header, "vp": vp, "vp_tk": vp_tk, "info": lbl_info}

    def _bara_status(self):
        sb = ctk.CTkFrame(self, fg_color="#16161a", corner_radius=0, height=30)
        sb.pack(side="bottom", fill="x")
        sb.pack_propagate(False)
        sb.grid_columnconfigure(0, weight=1)

        self._sb_stanga = ctk.CTkLabel(
            sb, text="●  Niciun fișier selectat.", text_color=CLR_DIM,
            font=ctk.CTkFont(family="Courier New", size=11), anchor="w"
        )
        self._sb_stanga.grid(row=0, column=0, padx=12, sticky="w")

        sb_dr = ctk.CTkFrame(sb, fg_color="transparent")
        sb_dr.grid(row=0, column=1, padx=12, sticky="e")

        ctk.CTkLabel(
            sb_dr,
            text=f"model: {os.path.basename(MODEL_PATH)}",
            text_color=CLR_DIM,
            font=ctk.CTkFont(family="Courier New", size=11)
        ).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(
            sb_dr, text="dark", text_color=CLR_ACCENT,
            font=ctk.CTkFont(family="Courier New", size=11)
        ).pack(side="left")

    def _initializeaza_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Model lipsă",
                f"Fișierul modelului nu a fost găsit:\n{MODEL_PATH}\n\n"
                "Pune best_model.pt lângă app.py și repornește aplicația."
            )
            return
        try:
            self._model = incarca_model(DEVICE)
        except Exception as exc:
            messagebox.showerror("Eroare încărcare model", str(exc))

    def _deschide_fisier(self):
        cale = filedialog.askopenfilename(
            title="Selectează fișier .npy",
            filetypes=[("NumPy files", "*.npy")]
        )
        if not cale:
            return
        try:
            img2d = citeste_imagine_npy(cale)
        except Exception as exc:
            messagebox.showerror("Eroare încărcare imagine", str(exc))
            return

        self._cale_imagine = cale
        self._imagine_2d   = img2d

        nume = os.path.basename(cale)
        h, w = img2d.shape
        self._dot_status.configure(text_color=CLR_OK)
        self._txt_status.configure(text=f"{nume} încărcat")
        self._sb_stanga.configure(text=f"●  {nume}   dim: {w} × {h}")
        self._pan_st["info"].configure(text=f"{w} × {h}")
        self._btn_segment.configure(state="normal")

        self._redeseaza_stanga()

    def _ruleaza_segmentare(self):
        if self._imagine_2d is None:
            messagebox.showerror("Eroare", "Selectează mai întâi o imagine .npy.")
            return
        if self._model is None:
            messagebox.showerror("Eroare", "Modelul nu este încărcat. Verifică best_model.pt.")
            return
        self._afiseaza_incarcare(True)
        self._btn_segment.configure(state="disabled")
        self._btn_deschide.configure(state="disabled")

        def task():
            try:
                proc = preprocesare(self._imagine_2d)
                rez  = inferenta(self._model, proc, DEVICE)
            except Exception as exc:
                self.after(0, lambda: self._eroare_segmentare(str(exc)))
                return
            self.after(0, lambda: self._segmentare_finalizata(rez))

        threading.Thread(target=task, daemon=True).start()

    def _eroare_segmentare(self, msg):
        self._afiseaza_incarcare(False)
        self._btn_segment.configure(state="normal")
        self._btn_deschide.configure(state="normal")
        messagebox.showerror("Eroare inferență", msg)

    def _segmentare_finalizata(self, pred):
        self._pred = pred
        self._afiseaza_incarcare(False)
        self._btn_segment.configure(state="normal")
        self._btn_deschide.configure(state="normal")
        self._btn_overlay.configure(state="normal")
        self._slider.configure(state="normal")

        n_active = sum(1 for i, n in enumerate(CLASE) if pred[i].max() > 0)
        self._dot_status.configure(text_color=CLR_ACCENT)
        self._txt_status.configure(text=f"{n_active} organe segmentate")
        self._pan_dr["info"].configure(text=f"{len(CLASE)} organe")

        self._redeseaza_dreapta()
        if self._overlay_activ:
            self._redeseaza_stanga()

    def _comuta_overlay(self):
        self._overlay_activ = not self._overlay_activ
        if self._overlay_activ:
            self._btn_overlay.configure(fg_color="#4c2d99", text_color="#c9b6ff")
        else:
            self._btn_overlay.configure(fg_color=BG_RAISED, text_color=CLR_SUBTLE)
        self._redeseaza_stanga()

    def _comuta_organ(self, cls):
        self._organ_viz[cls] = not self._organ_viz[cls]
        chip = self._chips[cls]
        chip["active"] = self._organ_viz[cls]
        if chip["active"]:
            chip["frame"].configure(fg_color=BG_CARD)
            chip["swatch"].configure(fg_color=COLORS[cls])
            chip["label"].configure(text_color=CLR_TEXT)
        else:
            chip["frame"].configure(fg_color=BG_RAISED)
            chip["swatch"].configure(fg_color=CLR_DIM)
            chip["label"].configure(text_color=CLR_DIM)
        if self._pred is not None:
            self._redeseaza_dreapta()
            if self._overlay_activ:
                self._redeseaza_stanga()

    def _schimba_opacitate(self, val):
        self._opacitate = float(val) / 100.0
        self._lbl_opacitate.configure(text=f"{int(val)}%")
        if self._overlay_activ and self._pred is not None:
            self._redeseaza_stanga()

    def _afiseaza_incarcare(self, arata):
        if arata:
            self._ph_dr.place_forget()
            if self._canvas_dr is not None:
                self._canvas_dr.get_tk_widget().place_forget()
            self._frame_incarcare.place(relx=0.5, rely=0.5, anchor="center")
            self._bara_incarcare.start()
        else:
            self._bara_incarcare.stop()
            self._frame_incarcare.place_forget()

    def _redeseaza_stanga(self):
        if self._imagine_2d is None:
            return

        vp = self._pan_st["vp_tk"]

        if self._fig_st is None:
            self._fig_st, self._ax_st = plt.subplots(1, 1)
            self._fig_st.patch.set_facecolor(BG_VIEW)
            self._canvas_st = FigureCanvasTkAgg(self._fig_st, master=vp)
            self._canvas_st.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._ph_st.place_forget()

        ax = self._ax_st
        ax.cla()
        ax.set_facecolor(BG_VIEW)
        ax.imshow(self._imagine_2d, cmap='gray', interpolation='nearest')

        if self._overlay_activ and self._pred is not None:
            for i, cls in enumerate(CLASE):
                if not self._organ_viz[cls]:
                    continue
                masca = self._pred[i]
                if masca.max() == 0:
                    continue
                r, g, b = COLORS_RGB[cls]
                rgba = np.zeros((*masca.shape, 4), dtype=np.float32)
                rgba[..., 0] = r
                rgba[..., 1] = g
                rgba[..., 2] = b
                rgba[..., 3] = masca * self._opacitate
                ax.imshow(rgba, interpolation='nearest')

            handles = [
                Patch(facecolor=COLORS_RGB[n], label=n.capitalize())
                for n in CLASE if self._organ_viz[n]
            ]
            if handles:
                ax.legend(
                    handles=handles, loc='upper right',
                    framealpha=0.65, facecolor=BG_CARD,
                    labelcolor=CLR_TEXT, fontsize=9, handlelength=1.2,
                    edgecolor=CLR_BORDER
                )

        titlu = "Overlay segmentare" if self._overlay_activ else "Imagine originală"
        ax.set_title(titlu, color=CLR_TEXT, fontsize=11, pad=6)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_edgecolor(CLR_BORDER)

        self._fig_st.tight_layout(pad=0.4)
        self._canvas_st.draw()

    def _redeseaza_dreapta(self):
        if self._pred is None:
            return

        vp = self._pan_dr["vp_tk"]

        if self._fig_dr is None:
            self._fig_dr, axe = plt.subplots(2, 2)
            self._axe_dr = axe.flatten()
            self._fig_dr.patch.set_facecolor(BG_VIEW)
            self._canvas_dr = FigureCanvasTkAgg(self._fig_dr, master=vp)
            self._canvas_dr.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._ph_dr.place_forget()

        for i, ax in enumerate(self._axe_dr):
            cls = CLASE[i]
            ax.cla()
            ax.set_facecolor(BG_VIEW)

            if self._organ_viz[cls]:
                ax.imshow(self._pred[i], cmap='gray', interpolation='nearest')
                culoare_titlu = COLORS[cls]
            else:
                ax.imshow(np.zeros_like(self._pred[i]), cmap='gray', interpolation='nearest')
                culoare_titlu = CLR_DIM

            ax.set_title(cls.capitalize(), color=culoare_titlu, fontsize=10, pad=4)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_edgecolor(CLR_BORDER)

        self._fig_dr.tight_layout(pad=0.6)
        self._canvas_dr.draw()


if __name__ == "__main__":
    app = App()
    app.mainloop()
