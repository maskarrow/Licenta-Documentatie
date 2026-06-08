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

WINDOW_BG = "#1b1b20"
FRAME_BG = "#232329"
CARD_BG = "#2a2a31"
ELEVATED = "#33333b"
BORDER = "#3a3a44"
TEXT = "#e8e8ee"
TEXT_MUTED = "#8b8b97"
TEXT_FAINT = "#5f5f6b"
ACCENT = "#8b5cf6"
ACCENT_HOVER = "#7c4dfb"
OK_GREEN = "#22c55e"
VIEWPORT_BG = "#0e0e10"

ORGAN_NAMES = ['artery', 'liver', 'stomach', 'vein']
NUM_CLASSES = 4

ORGAN_COLORS_HEX = {
    'artery': "#f59e0b",
    'liver': "#ef4444",
    'stomach': "#22c55e",
    'vein': "#3b82f6",
}

def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

ORGAN_COLORS_RGB = {k: _hex_to_rgb(v) for k, v in ORGAN_COLORS_HEX.items()}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_model.pt")

val_transform = A.Compose([
    A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
    ToTensorV2()
])


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
        resnet = models.resnet50(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4
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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        d4 = self.dec4(e5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.final_conv(d1)


def to_uint8(image):
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    image -= image.min()
    if image.max() > 0:
        image = image / image.max() * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def mean_filter(image, radius=3):
    image = image.astype(np.float32)
    padded = np.pad(image, radius, mode="reflect")
    s = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float32)
    s[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    H, W = image.shape
    a = s[2*radius+1 : 2*radius+1+H, 2*radius+1 : 2*radius+1+W]
    b = s[:H, 2*radius+1 : 2*radius+1+W]
    c = s[2*radius+1 : 2*radius+1+H, :W]
    d = s[:H, :W]
    return (a - b - c + d) / (2*radius+1)**2


def binary_dilate(mask, radius=1):
    mask = mask.astype(bool)
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    for dy in range(2*radius+1):
        for dx in range(2*radius+1):
            out |= padded[dy:dy+mask.shape[0], dx:dx+mask.shape[1]]
    return out


def binary_erode(mask, radius=1):
    mask = mask.astype(bool)
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.ones_like(mask)
    for dy in range(2*radius+1):
        for dx in range(2*radius+1):
            out &= padded[dy:dy+mask.shape[0], dx:dx+mask.shape[1]]
    return out


def remove_small_objects(mask, radius=1):
    return binary_dilate(binary_erode(mask, radius), radius)


def clean_text(image):
    image = to_uint8(image)
    local_mean = mean_filter(image, radius=10)
    text_mask = (image >= local_mean + 22) | (image <= local_mean - 22)
    text_mask = remove_small_objects(text_mask, radius=2)
    text_mask = binary_dilate(text_mask, radius=3)
    background = mean_filter(image, radius=12).astype(np.uint8)
    clean = image.copy()
    clean[text_mask] = background[text_mask]
    return clean


def preprocess_for_inference(image):
    clean = clean_text(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(clean)


def load_model(device):
    model = MultiOutputUNet(NUM_CLASSES).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_npy_image(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == () and data.dtype == object:
        data = data.item()
    if isinstance(data, dict):
        if 'image' not in data:
            raise ValueError("Dict-ul .npy nu contine cheia 'image'.")
        image = data['image']
    else:
        image = data
    image = np.asarray(image).astype(np.float32)
    if image.ndim == 3 and image.shape[-1] == 3:
        image = image[:, :, 0]
    if image.ndim == 3:
        image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"Imaginea trebuie sa fie 2D dupa procesare, am primit shape {image.shape}.")
    return image


def run_inference(model, image_2d, device):
    augmented = val_transform(image=image_2d)
    tensor = augmented['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        preds = (torch.sigmoid(logits) > 0.5).float()
    return preds[0].cpu().numpy()


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Segmentare Organe")
        self.geometry("1280x820")
        self.minsize(1200, 760)
        self.configure(fg_color=WINDOW_BG)

        self._image_path = None
        self._image_2d = None
        self._preds = None
        self._model = None
        self._overlay_on = False
        self._opacity = 0.70
        self._organ_vis = {n: True for n in ORGAN_NAMES}

        self._left_fig = self._left_ax = self._left_canvas = None
        self._right_fig = self._right_axes = self._right_canvas = None

        self._build_ui()
        self._try_load_model()

    def _build_ui(self):
        self._build_statusbar()
        self._build_toolbar()
        self._build_legend()
        self._build_panels()

    def _build_toolbar(self):
        tb = ctk.CTkFrame(self, fg_color=FRAME_BG, corner_radius=0, height=62)
        tb.pack(side="top", fill="x")
        tb.pack_propagate(False)
        tb.grid_columnconfigure(7, weight=1)

        btn_font = ctk.CTkFont(size=13, weight="bold")

        self._btn_browse = ctk.CTkButton(
            tb, text="Browse Image", width=148, height=38,
            corner_radius=9, fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=TEXT, font=btn_font, command=self._browse
        )
        self._btn_browse.grid(row=0, column=0, padx=(16, 6), pady=12)

        self._btn_segment = ctk.CTkButton(
            tb, text="Segment", width=120, height=38,
            corner_radius=9, fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=TEXT, font=btn_font, state="disabled",
            command=self._segment
        )
        self._btn_segment.grid(row=0, column=1, padx=6, pady=12)

        self._btn_overlay = ctk.CTkButton(
            tb, text="Overlay View", width=138, height=38,
            corner_radius=9, fg_color=ELEVATED, hover_color="#3d3d46",
            text_color=TEXT_MUTED, font=ctk.CTkFont(size=13),
            state="disabled", command=self._toggle_overlay
        )
        self._btn_overlay.grid(row=0, column=2, padx=6, pady=12)

        ctk.CTkFrame(tb, width=1, height=26, fg_color=BORDER).grid(
            row=0, column=3, padx=12, pady=18
        )

        ctk.CTkLabel(
            tb, text="Opacitate măști", text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=12)
        ).grid(row=0, column=4, padx=(0, 4), pady=12)

        self._slider = ctk.CTkSlider(
            tb, from_=0, to=100, number_of_steps=100, width=140,
            progress_color=ACCENT, button_color=TEXT,
            button_hover_color=ACCENT, state="disabled",
            command=self._on_opacity
        )
        self._slider.set(70)
        self._slider.grid(row=0, column=5, padx=4, pady=12)

        self._opacity_lbl = ctk.CTkLabel(
            tb, text="70%", text_color=TEXT_MUTED, width=38,
            font=ctk.CTkFont(family="Courier New", size=12)
        )
        self._opacity_lbl.grid(row=0, column=6, padx=(2, 10), pady=12)

        ctk.CTkFrame(tb, fg_color="transparent").grid(row=0, column=7, sticky="ew")

        pill = ctk.CTkFrame(tb, fg_color=CARD_BG, corner_radius=8, height=34)
        pill.grid(row=0, column=8, padx=(0, 16), pady=14)

        self._pill_dot = ctk.CTkLabel(
            pill, text="●", text_color=TEXT_FAINT,
            font=ctk.CTkFont(size=10), width=18
        )
        self._pill_dot.pack(side="left", padx=(8, 2), pady=4)

        self._pill_txt = ctk.CTkLabel(
            pill, text="Niciun fișier selectat.",
            text_color=TEXT_MUTED, font=ctk.CTkFont(size=12), width=210
        )
        self._pill_txt.pack(side="left", padx=(0, 10), pady=4)

    def _build_legend(self):
        leg = ctk.CTkFrame(self, fg_color=WINDOW_BG, height=48)
        leg.pack(side="top", fill="x", padx=16, pady=(8, 0))
        leg.pack_propagate(False)

        ctk.CTkLabel(
            leg, text="ORGANE", text_color=TEXT_FAINT,
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(side="left", padx=(0, 14))

        self._chips = {}
        for name in ORGAN_NAMES:
            chip_data = self._make_chip(leg, name)
            chip_data["frame"].pack(side="left", padx=5)
            self._chips[name] = chip_data

    def _make_chip(self, parent, name):
        color = ORGAN_COLORS_HEX[name]
        frame = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=15, height=32)
        frame.pack_propagate(False)

        swatch = ctk.CTkLabel(frame, text="", fg_color=color, corner_radius=3, width=14, height=14)
        swatch.pack(side="left", padx=(10, 5), pady=9)

        label = ctk.CTkLabel(
            frame, text=name.capitalize(),
            text_color=TEXT, font=ctk.CTkFont(size=12)
        )
        label.pack(side="left", padx=(0, 12), pady=9)

        def _toggle(event=None, n=name):
            self._toggle_organ(n)

        for w in (frame, swatch, label):
            w.bind("<Button-1>", _toggle)
            w.configure(cursor="hand2")

        return {"frame": frame, "swatch": swatch, "label": label, "active": True}

    def _build_panels(self):
        container = ctk.CTkFrame(self, fg_color=WINDOW_BG)
        container.pack(side="top", fill="both", expand=True, padx=16, pady=10)
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)
        container.grid_rowconfigure(0, weight=1)

        self._lp = self._make_panel(container, "Imagine originală", col=0)
        self._rp = self._make_panel(container, "Măști segmentare", col=1)

        self._left_ph = tk.Label(
            self._lp["vp_tk"], text="Selectează un fișier .npy",
            fg=TEXT_FAINT, bg=VIEWPORT_BG, font=("Segoe UI", 11)
        )
        self._left_ph.place(relx=0.5, rely=0.5, anchor="center")

        self._right_ph = tk.Label(
            self._rp["vp_tk"], text="Apasă Segment pentru a vedea măștile",
            fg=TEXT_FAINT, bg=VIEWPORT_BG, font=("Segoe UI", 11)
        )
        self._right_ph.place(relx=0.5, rely=0.5, anchor="center")

        self._loading_frame = tk.Frame(self._rp["vp_tk"], bg=VIEWPORT_BG)
        self._loading_bar = ctk.CTkProgressBar(
            self._loading_frame, mode="indeterminate",
            progress_color=ACCENT, width=200, height=6
        )
        self._loading_bar.pack()
        tk.Label(
            self._loading_frame, text="Segmentare în curs…",
            fg=TEXT_MUTED, bg=VIEWPORT_BG, font=("Segoe UI", 11)
        ).pack(pady=(10, 0))

    def _make_panel(self, parent, title, col):
        padx = (0, 7) if col == 0 else (7, 0)
        frame = ctk.CTkFrame(
            parent, fg_color=FRAME_BG, corner_radius=12,
            border_width=1, border_color=BORDER
        )
        frame.grid(row=0, column=col, sticky="nsew", padx=padx)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(frame, fg_color=CARD_BG, corner_radius=0, height=42)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text=title, text_color=TEXT,
            font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
        ).grid(row=0, column=0, padx=14, sticky="w", pady=10)

        info_lbl = ctk.CTkLabel(
            header, text="—", text_color=TEXT_FAINT,
            font=ctk.CTkFont(family="Courier New", size=11), anchor="e"
        )
        info_lbl.grid(row=0, column=1, padx=14, sticky="e", pady=10)

        vp = ctk.CTkFrame(frame, fg_color=VIEWPORT_BG, corner_radius=0)
        vp.grid(row=1, column=0, sticky="nsew")
        vp.grid_columnconfigure(0, weight=1)
        vp.grid_rowconfigure(0, weight=1)

        vp_tk = tk.Frame(vp, bg=VIEWPORT_BG)
        vp_tk.grid(row=0, column=0, sticky="nsew")

        return {"frame": frame, "header": header, "vp": vp, "vp_tk": vp_tk, "info": info_lbl}

    def _build_statusbar(self):
        sb = ctk.CTkFrame(self, fg_color="#16161a", corner_radius=0, height=30)
        sb.pack(side="bottom", fill="x")
        sb.pack_propagate(False)
        sb.grid_columnconfigure(0, weight=1)

        self._sb_left = ctk.CTkLabel(
            sb, text="●  Niciun fișier selectat.", text_color=TEXT_FAINT,
            font=ctk.CTkFont(family="Courier New", size=11), anchor="w"
        )
        self._sb_left.grid(row=0, column=0, padx=12, sticky="w")

        right_sb = ctk.CTkFrame(sb, fg_color="transparent")
        right_sb.grid(row=0, column=1, padx=12, sticky="e")

        ctk.CTkLabel(
            right_sb,
            text=f"model: {os.path.basename(MODEL_PATH)}",
            text_color=TEXT_FAINT,
            font=ctk.CTkFont(family="Courier New", size=11)
        ).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(
            right_sb, text="dark", text_color=ACCENT,
            font=ctk.CTkFont(family="Courier New", size=11)
        ).pack(side="left")

    def _try_load_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Model lipsă",
                f"Fișierul modelului nu a fost găsit:\n{MODEL_PATH}\n\n"
                "Pune best_model.pt lângă app.py și repornește aplicația."
            )
            return
        try:
            self._model = load_model(DEVICE)
        except Exception as exc:
            messagebox.showerror("Eroare încărcare model", str(exc))

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Selectează fișier .npy",
            filetypes=[("NumPy files", "*.npy")]
        )
        if not path:
            return
        try:
            image_2d = load_npy_image(path)
        except Exception as exc:
            messagebox.showerror("Eroare încărcare imagine", str(exc))
            return

        self._image_path = path
        self._image_2d = image_2d

        name = os.path.basename(path)
        h, w = image_2d.shape
        self._pill_dot.configure(text_color=OK_GREEN)
        self._pill_txt.configure(text=f"{name} încărcat")
        self._sb_left.configure(text=f"●  {name}   dim: {w} × {h}")
        self._lp["info"].configure(text=f"{w} × {h}")
        self._btn_segment.configure(state="normal")

        self._redraw_left()

    def _segment(self):
        if self._image_2d is None:
            messagebox.showerror("Eroare", "Selectează mai întâi o imagine .npy.")
            return
        if self._model is None:
            messagebox.showerror("Eroare", "Modelul nu este încărcat. Verifică best_model.pt.")
            return
        self._show_loading(True)
        self._btn_segment.configure(state="disabled")
        self._btn_browse.configure(state="disabled")

        def worker():
            try:
                preprocessed = preprocess_for_inference(self._image_2d)
                preds = run_inference(self._model, preprocessed, DEVICE)
            except Exception as exc:
                self.after(0, lambda: self._on_segment_error(str(exc)))
                return
            self.after(0, lambda: self._on_segment_done(preds))

        threading.Thread(target=worker, daemon=True).start()

    def _on_segment_error(self, msg):
        self._show_loading(False)
        self._btn_segment.configure(state="normal")
        self._btn_browse.configure(state="normal")
        messagebox.showerror("Eroare inferență", msg)

    def _on_segment_done(self, preds):
        self._preds = preds
        self._show_loading(False)
        self._btn_segment.configure(state="normal")
        self._btn_browse.configure(state="normal")
        self._btn_overlay.configure(state="normal")
        self._slider.configure(state="normal")

        n_active = sum(1 for i, n in enumerate(ORGAN_NAMES) if preds[i].max() > 0)
        self._pill_dot.configure(text_color=ACCENT)
        self._pill_txt.configure(text=f"{n_active} organe segmentate")
        self._rp["info"].configure(text=f"{len(ORGAN_NAMES)} organe")

        self._redraw_right()
        if self._overlay_on:
            self._redraw_left()

    def _toggle_overlay(self):
        self._overlay_on = not self._overlay_on
        if self._overlay_on:
            self._btn_overlay.configure(fg_color="#4c2d99", text_color="#c9b6ff")
        else:
            self._btn_overlay.configure(fg_color=ELEVATED, text_color=TEXT_MUTED)
        self._redraw_left()

    def _toggle_organ(self, name):
        self._organ_vis[name] = not self._organ_vis[name]
        chip = self._chips[name]
        chip["active"] = self._organ_vis[name]
        if chip["active"]:
            chip["frame"].configure(fg_color=CARD_BG)
            chip["swatch"].configure(fg_color=ORGAN_COLORS_HEX[name])
            chip["label"].configure(text_color=TEXT)
        else:
            chip["frame"].configure(fg_color=ELEVATED)
            chip["swatch"].configure(fg_color=TEXT_FAINT)
            chip["label"].configure(text_color=TEXT_FAINT)
        if self._preds is not None:
            self._redraw_right()
            if self._overlay_on:
                self._redraw_left()

    def _on_opacity(self, value):
        self._opacity = float(value) / 100.0
        self._opacity_lbl.configure(text=f"{int(value)}%")
        if self._overlay_on and self._preds is not None:
            self._redraw_left()

    def _show_loading(self, show):
        if show:
            self._right_ph.place_forget()
            if self._right_canvas is not None:
                self._right_canvas.get_tk_widget().place_forget()
            self._loading_frame.place(relx=0.5, rely=0.5, anchor="center")
            self._loading_bar.start()
        else:
            self._loading_bar.stop()
            self._loading_frame.place_forget()

    def _redraw_left(self):
        if self._image_2d is None:
            return

        vp = self._lp["vp_tk"]

        if self._left_fig is None:
            self._left_fig, self._left_ax = plt.subplots(1, 1)
            self._left_fig.patch.set_facecolor(VIEWPORT_BG)
            self._left_canvas = FigureCanvasTkAgg(self._left_fig, master=vp)
            self._left_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._left_ph.place_forget()

        ax = self._left_ax
        ax.cla()
        ax.set_facecolor(VIEWPORT_BG)
        ax.imshow(self._image_2d, cmap='gray', interpolation='nearest')

        if self._overlay_on and self._preds is not None:
            for i, name in enumerate(ORGAN_NAMES):
                if not self._organ_vis[name]:
                    continue
                mask = self._preds[i]
                if mask.max() == 0:
                    continue
                r, g, b = ORGAN_COLORS_RGB[name]
                rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
                rgba[..., 0] = r
                rgba[..., 1] = g
                rgba[..., 2] = b
                rgba[..., 3] = mask * self._opacity
                ax.imshow(rgba, interpolation='nearest')

            handles = [
                Patch(facecolor=ORGAN_COLORS_RGB[n], label=n.capitalize())
                for n in ORGAN_NAMES if self._organ_vis[n]
            ]
            if handles:
                ax.legend(
                    handles=handles, loc='upper right',
                    framealpha=0.65, facecolor=CARD_BG,
                    labelcolor=TEXT, fontsize=9, handlelength=1.2,
                    edgecolor=BORDER
                )

        title = "Overlay segmentare" if self._overlay_on else "Imagine originală"
        ax.set_title(title, color=TEXT, fontsize=11, pad=6)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

        self._left_fig.tight_layout(pad=0.4)
        self._left_canvas.draw()

    def _redraw_right(self):
        if self._preds is None:
            return

        vp = self._rp["vp_tk"]

        if self._right_fig is None:
            self._right_fig, axes = plt.subplots(2, 2)
            self._right_axes = axes.flatten()
            self._right_fig.patch.set_facecolor(VIEWPORT_BG)
            self._right_canvas = FigureCanvasTkAgg(self._right_fig, master=vp)
            self._right_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._right_ph.place_forget()

        for i, ax in enumerate(self._right_axes):
            name = ORGAN_NAMES[i]
            ax.cla()
            ax.set_facecolor(VIEWPORT_BG)

            if self._organ_vis[name]:
                ax.imshow(self._preds[i], cmap='gray', interpolation='nearest')
                title_color = ORGAN_COLORS_HEX[name]
            else:
                ax.imshow(np.zeros_like(self._preds[i]), cmap='gray', interpolation='nearest')
                title_color = TEXT_FAINT

            ax.set_title(name.capitalize(), color=title_color, fontsize=10, pad=4)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)

        self._right_fig.tight_layout(pad=0.6)
        self._right_canvas.draw()


if __name__ == "__main__":
    app = App()
    app.mainloop()
