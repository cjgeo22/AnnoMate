#!/usr/bin/env python3
"""
AnnoMate – an improved drawing and inference tool
=================================================

This module defines a modernised version of the AnnoMate application for
image annotation and model inference.  It combines an interactive
annotation canvas with a lightweight Grad‑CAM viewer to support defect
labelling workflows.  The code has been refactored for clarity and
maintainability and no longer relies on an external ``style_config``.  All
visual styling is contained in a few constants at the top of the file.

Key features
------------

* Draw on images using multiple coloured pens corresponding to defect
  categories (chip, scratch, gouge, inclusion, void, other).
* Toggle between freehand drawing and flood‑fill modes on the current pen.
* Navigate through a folder of images, undo/redo strokes and clear
  annotations.
* Save individual masks for each selected defect as well as a combined
  overlay; export a summary of your work to CSV or Excel.
* Run a pretrained ResNet on the current image to classify it as
  ``Accept`` or ``Reject`` and visualise the model’s attention via
  Grad‑CAM.
* Automatically update the inference view whenever the active image or
  overlay changes.

This implementation strives to remain self‑contained: all configuration
files, models and intermediate outputs live in the working directory so
that the tool can be packaged with PyInstaller or executed directly.

"""

from __future__ import annotations

import csv
import datetime
import glob
import json
import logging
import os
import sys
import configparser
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import models, transforms

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog


# ---------------------------------------------------------------------------
# Styling constants
#
# Modify these values to change the appearance of the UI without touching
# business logic.  Colours are specified in hexadecimal RGB notation, fonts
# are tuples of (family, size, optional style).
# ---------------------------------------------------------------------------

# Relative weight of the annotation canvas versus the control panel when
# resizing the main window.  Larger numbers give more space to the canvas.
COLUMN_WEIGHTS: Dict[int, int] = {0: 4, 1: 2}
ROW_WEIGHTS: Dict[int, int] = {0: 1}

# Padding (in pixels) used around UI widgets
PAD_SMALL: int = 4
PAD_MEDIUM: int = 8
PAD_LARGE: int = 12

# Font definitions
FONT_FAMILY: str = "Segoe UI"
BUTTON_FONT: Tuple[str, int] = (FONT_FAMILY, 10)
LABEL_FONT: Tuple[str, int] = (FONT_FAMILY, 10)
ENTRY_FONT: Tuple[str, int] = (FONT_FAMILY, 10)

# Additional padding for buttons (horizontal, vertical)
BUTTON_PAD: Tuple[int, int] = (4, 2)

# General colour scheme
BACKGROUND: str = "#f4f4f4"
CANVAS_BG: str = "#2e2e2e"
CANVAS_CURSOR: str = "crosshair"

# Timestamp formats
TS_DISPLAY_FMT: str = "%Y-%m-%d %H:%M:%S"
TS_FILENAME_FMT: str = "%Y%m%d_%H%M%S"

# Colours for each pen (RGBA).  These colours map onto defect labels via
# ``DEFECT_TO_PEN`` defined further below.
PEN_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    "pen1": (230, 25, 75, 255),  # red
    "pen2": (60, 180, 75, 255),  # green
    "pen3": (0, 130, 200, 255),  # blue
    "pen4": (245, 130, 48, 255),  # orange
    "pen5": (145, 30, 180, 255),  # purple
    "pen6": (70, 240, 240, 255),  # cyan
}

# Tolerance when comparing colours during masking.  Two colours are
# considered equal if each channel differs by less than or equal to this
# value.
COLOR_TOLERANCE: int = 10

# Supported image file extensions
EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Mapping from defect names to pen identifiers.  Adjust these to alter
# which pen colour is used for a particular defect category.
DEFECT_TO_PEN: Dict[str, str] = {
    "chip": "pen1",
    "scratch": "pen2",
    "gouge": "pen3",
    "inclusion": "pen4",
    "void": "pen5",
    "other": "pen6",
}

# Reverse lookup from pen back to defect.  This dictionary is derived
# automatically from ``DEFECT_TO_PEN`` and is provided for convenience.
PEN_TO_DEFECT: Dict[str, str] = {v: k for k, v in DEFECT_TO_PEN.items()}


# ---------------------------------------------------------------------------
# Configuration and logging
# ---------------------------------------------------------------------------

# Path to the user configuration file.  The configuration stores the last
# selected input folder, defects JSON file and output subdirectory.  When
# the file does not exist a default configuration is created.
CONFIG_PATH: str = os.path.join(os.path.dirname(__file__), "config.ini")
config = configparser.ConfigParser()
if not os.path.exists(CONFIG_PATH):
    config["Paths"] = {
        "input_folder": "",
        "defects_config": "defects_config.json",
        "output_subdir": "output",
    }
    with open(CONFIG_PATH, "w") as cfg_file:
        config.write(cfg_file)
else:
    config.read(CONFIG_PATH)


def save_config() -> None:
    """Persist the global configuration to disk."""
    with open(CONFIG_PATH, "w") as cfg_file:
        config.write(cfg_file)


# Configure application‑wide logging.  A rotating file handler prevents
# unbounded log growth.  Additional handlers are added later once the
# output directory is known (see MaskingTool.__init__).
logger = logging.getLogger("AnnoMate")
logger.setLevel(logging.DEBUG)
_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_file_handler = RotatingFileHandler(
    "AnnoMate.log", maxBytes=5 * 1024 * 1024, backupCount=2
)
_file_handler.setFormatter(_formatter)
logger.addHandler(_file_handler)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def resource_path(relative_path: str) -> str:
    """
    Resolve resource paths both in a development environment and when bundled
    with PyInstaller.  If the application is frozen, resources are located
    alongside the executable in a temporary directory; otherwise they live
    relative to this source file.

    Parameters
    ----------
    relative_path: str
        Relative path to a resource within the project.

    Returns
    -------
    str
        Absolute path pointing to the resource.
    """
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)


# ---------------------------------------------------------------------------
# Grad‑CAM implementation
# ---------------------------------------------------------------------------


class GradCAM:
    """
    Compute class activation maps for a PyTorch model.  When called on a
    preprocessed tensor and an optional class index, it returns a 2‑D
    normalised heatmap highlighting the regions most relevant to that class.

    Attributes
    ----------
    model: nn.Module
        The model in evaluation mode.  Gradients are captured from the
        specified target layer.
    target_layer: nn.Module
        The convolutional layer whose activations and gradients are used
        for computing the CAM.
    gradients: Optional[torch.Tensor]
        Saved gradients from the backward pass.
    activations: Optional[torch.Tensor]
        Saved activations from the forward pass.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Attach forward and backward hooks to the target layer."""

        def forward_hook(
            _: nn.Module, __: Tuple[torch.Tensor], output: torch.Tensor
        ) -> None:
            self.activations = output.detach()

        def backward_hook(
            _: nn.Module, grad_in: Tuple[torch.Tensor], grad_out: Tuple[torch.Tensor]
        ) -> None:
            # Capture the gradient with respect to the layer's output
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    @staticmethod
    def _normalize(cam: torch.Tensor) -> torch.Tensor:
        """Normalize a CAM to the range [0, 1] to aid visualisation."""
        cam_min = cam.min()
        cam = cam - cam_min
        cam_max = cam.max().clamp(min=1e-8)
        cam = cam / cam_max
        return cam

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Compute a CAM for the supplied tensor.  If no class index is
        provided the one with the highest logit is chosen.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (1, C, H, W) after preprocessing.
        class_idx: int, optional
            Index of the class to compute the map for.  If ``None``, the
            class with maximum logit is used.

        Returns
        -------
        np.ndarray
            A 2‑D array of floats normalised to [0, 1].
        """
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Zero existing gradients and backpropagate only the selected logit
        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward(retain_graph=True)

        # Spatially average the gradients to obtain weights for each channel
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = nn.functional.interpolate(
            cam, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        cam = self._normalize(cam.squeeze())
        return cam.cpu().numpy()


def transforms_eval() -> transforms.Compose:
    """
    Preprocessing pipeline for inference.  Resize images to 224×224,
    convert them to tensors and normalise with ImageNet statistics.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def overlay(cam: np.ndarray, img_rgb: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Blend a CAM with the original image to create a heatmap overlay.

    Parameters
    ----------
    cam: np.ndarray
        Normalised class activation map.
    img_rgb: np.ndarray
        Original image as a H×W×3 array in RGB format.
    alpha: float
        Weighting factor for the heatmap.  Larger values emphasise the
        CAM while smaller values retain more of the original image.

    Returns
    -------
    np.ndarray
        The blended image.
    """
    heat = np.uint8(255 * cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0])))
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, heat, alpha, 0)


def build_resnet(arch: str) -> nn.Module:
    """
    Construct a ResNet model without pretrained weights.  The final
    fully‑connected layer is replaced to output two classes.

    Parameters
    ----------
    arch: str
        One of ``'resnet18'``, ``'resnet50'``, ``'resnet101'``, ``'resnet152'``.

    Returns
    -------
    nn.Module
        A fresh ResNet model with a two‑class output layer.
    """
    fns = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }
    if arch not in fns:
        raise ValueError(f"Unsupported arch '{arch}'.")
    model = fns[arch](weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def guess_arch(state: Dict[str, torch.Tensor]) -> str:
    """
    Heuristically determine which ResNet architecture a checkpoint
    corresponds to based on its parameter shapes.

    Parameters
    ----------
    state: Dict[str, torch.Tensor]
        Mapping from parameter names to tensors from a checkpoint.

    Returns
    -------
    str
        The name of the architecture, e.g. ``'resnet18'``.
    """
    fc_in = state["fc.weight"].shape[1]
    if fc_in == 512:
        return "resnet18"
    blocks_l3 = len({k.split(".")[1] for k in state if k.startswith("layer3.")})
    return {6: "resnet50", 23: "resnet101", 36: "resnet152"}.get(blocks_l3, "unknown")


def param_stats(model: nn.Module) -> Tuple[int, int]:
    """Return the total and trainable parameter counts of a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_inference(
    model_path: str, image_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, float]:
    """
    Load a checkpoint, run inference on a single image and compute the
    Grad‑CAM heatmap.  A textual summary of the run and prediction
    statistics is assembled for display.

    Parameters
    ----------
    model_path: str
        Path to a saved PyTorch checkpoint (.pth or .pt file).  Either a
        ``state_dict`` or a checkpoint with a ``state_dict`` field is
        accepted.
    image_path: str
        Path to the image to run inference on.

    Returns
    -------
    Tuple containing:
        - The original image as a NumPy array (H×W×3).
        - The heatmap as a NumPy array (H×W×3).
        - The overlay image as a NumPy array (H×W×3).
        - A human‑readable summary string describing the model and its
          prediction.
        - The predicted class label (``'Accept'`` or ``'Reject'``).
        - The confidence (probability) of the predicted class.
    """
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        arch = ckpt.get("arch") or guess_arch(ckpt["state_dict"])
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
        arch = guess_arch(state_dict)

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = build_resnet(arch).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)
    x = transforms_eval()(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    idx = int(probs.argmax())
    classes = ["Accept", "Reject"]
    pred = classes[idx]

    # Generate Grad‑CAM
    cam = GradCAM(model, model.layer4[-1].conv2)(x, class_idx=idx)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay_img = overlay(cam, img_np)

    total_p, trainable_p = param_stats(model)
    lines = [
        "== Inference Results ==",
        f"Image              : {image_path}",
        f"Model checkpoint   : {model_path}",
        f"Predicted class    : {pred}",
        f"Probabilities      : Accept={probs[0]:.4f} | Reject={probs[1]:.4f}",
        "",
        "== Model Details ==",
        f"Architecture       : {arch}",
        f"Total parameters   : {total_p:,}",
        f"Trainable params   : {trainable_p:,}",
        f"Final FC in-features: {model.fc.in_features}",
        f"Blocks per stage   : {[len(l) for l in (model.layer1, model.layer2, model.layer3, model.layer4)]} (layers1-4)",
        "",
        "Inference completed successfully.",
    ]
    summary = "\n".join(lines)

    return img_np, heatmap, overlay_img, summary, pred, float(probs[idx])


# ---------------------------------------------------------------------------
# Masking/Annotation tool
# ---------------------------------------------------------------------------


class MaskingTool:
    """
    Interactive tool for annotating images with defect masks.  Users can
    draw freehand strokes, fill regions, assign labels to strokes and
    export masks.  The tool maintains its own undo/redo history and
    supports dynamic resizing of the main window.

    Parameters
    ----------
    parent: tk.Widget
        The Tkinter parent in which the tool should embed itself.  Typically
        this is a frame inside a notebook tab.
    input_dir: str
        Directory containing images to annotate.
    output_dir: str
        Directory where masks, metadata and logs will be saved.
    config_path: str
        Path to a JSON file describing defect categories for each dataset.
    dataset: str
        Name of the dataset whose defect categories should be loaded from
        ``config_path``.
    """

    def __init__(
        self,
        parent: tk.Widget,
        input_dir: str,
        output_dir: str,
        config_path: str,
        dataset: str,
    ) -> None:
        self.root = parent  # For clarity; the main frame passed to this class
        # Ensure a single StringVar is used throughout the UI to reflect the
        # currently selected image.  This variable is shared between the
        # search bar and the metadata panel so that updates propagate to
        # both locations automatically.  See `_build_metadata_controls` for
        # usage.
        self.filename_var = tk.StringVar(master=self.root, value="")
        # Inference status fields displayed in the GUI
        self.model_status: str = ""
        self.model_confidence: float = 0.0
        # Default path to the model (can be changed via the settings menu).
        # Use `resource_path` to resolve the checkpoint relative to this
        # source file.  Without this, the model may not be found when
        # running from a different working directory and the default
        # checkpoint would silently fail to load.  Selecting a model
        # through the settings menu will update this value.
        self.model_path: str = resource_path("best_val_loss_model.pth")
        # If a model path has been persisted in the configuration file,
        # override the default.  When `model_path` is not set or is
        # empty, the default remains in effect.
        try:
            cfg_model_path = config["Paths"].get("model_path", "")
            if cfg_model_path:
                self.model_path = cfg_model_path
        except Exception:
            pass
        # Callback invoked whenever a new image is selected or the overlay
        # changes; set by the parent application
        self.on_image_change: Optional[Callable[[str, Image.Image], None]] = None

        self.input_dir = input_dir
        self.output_dir = output_dir
        self._zoom_after_id: Optional[str] = None
        # Placeholder layer so that self.layer.load() exists before any image
        self.layer: Image.Image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))

        # Ensure that the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Add a second log handler into the output directory.  Without this
        # the log file would only be written in the working directory.
        log_path = os.path.join(self.output_dir, "AnnoMate.log")
        fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=2)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_formatter)
        logger.addHandler(fh)

        # Load defect list from the JSON configuration.  Each dataset in the
        # configuration maps to a list of defect strings.  If the dataset
        # key is missing we fall back to the ``bowtie`` entry.
        cfg_path = resource_path(config_path)
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            self.defects: List[str] = cfg.get(dataset, cfg.get("bowtie", []))
        except Exception as ex:
            logger.error(f"Failed to load defects from {cfg_path}: {ex}")
            self.defects = []

        # Gather image paths from the input directory.  We only consider
        # files with recognised extensions.
        self.image_paths: List[str] = []
        for fname in os.listdir(input_dir):
            ext = os.path.splitext(fname.lower())[1]
            if ext in EXTENSIONS:
                self.image_paths.append(os.path.join(input_dir, fname))
        self.image_paths.sort()
        if not self.image_paths:
            raise RuntimeError(f"No images found in {input_dir}")

        # Cache predictions (original image, heatmap, overlay, summary, class, prob)
        self.pred_cache: Dict[
            str, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, float]]
        ] = {}
        # Show a progress bar while preloading predictions
        self._progress_var = tk.DoubleVar()
        self._progress_bar = ttk.Progressbar(
            self.root,
            orient="horizontal",
            length=300,
            mode="determinate",
            variable=self._progress_var,
        )
        self._progress_bar.pack(pady=10)
        self.root.update()
        self._preload_predictions()
        self._progress_bar.destroy()

        # Annotation state
        self.idx: int = 0  # index into image_paths
        self.mode: str = "draw"  # 'draw' or 'erase'
        self.history: List[Image.Image] = []  # undo history
        self.redo_stack: List[Image.Image] = []  # redo history
        self.pan_x: int = 0  # current pan offset in x
        self.pan_y: int = 0  # current pan offset in y
        self.zoom: float = 1.0  # current zoom factor
        self.drawing: bool = False  # whether the left mouse button is down

        # Current pen state
        self.current_pen: str = "pen1"
        self.pen_color: Tuple[int, int, int, int] = PEN_COLORS[self.current_pen]
        self.pen_width: int = 8
        self._tk_pen_color: str = "#%02x%02x%02x" % self.pen_color[:3]
        self._current_stroke_ids: List[int] = []

        # Unsaved clear flag used when navigating images; see prev_image/next_image
        self._unsaved_clear: bool = False

        # Build the user interface
        self._build_ui()

        # Bind loggers for clicks, key presses and window resizes
        self.root.bind_all("<ButtonPress>", self._log_button)
        self.root.bind_all(
            "<KeyPress>", lambda e: logger.info(f"Key {e.keysym} pressed in {e.widget}")
        )
        self.root.bind(
            "<Configure>",
            lambda e: (
                logger.info(f"Window resized to {e.width}x{e.height}")
                if e.widget == self.root
                else None
            ),
        )

        try:
            self.root.winfo_toplevel().protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            pass

        # Load the first image
        self._load_image()
        self.root.update()
        self._fit_image_to_canvas()
        self._show_image()
        # Trigger the initial inference callback if provided
        if self.on_image_change:
            self.on_image_change(self.image_paths[self.idx], self.layer)

    # ---------------------------------------------------------------------
    # UI construction methods
    # ---------------------------------------------------------------------

    def _build_styles(self) -> None:
        """Configure ttk styles for the entire application."""
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BACKGROUND)
        style.configure("TLabelFrame", background=BACKGROUND, font=LABEL_FONT)
        style.configure("TLabelFrame.Label", background=BACKGROUND, font=LABEL_FONT)
        style.configure("TLabel", background=BACKGROUND, font=LABEL_FONT)
        style.configure(
            "TButton", font=BUTTON_FONT, padding=BUTTON_PAD, foreground="#000000"
        )
        style.configure("TEntry", font=ENTRY_FONT)
        style.configure("TCheckbutton", background=BACKGROUND, font=LABEL_FONT)
        style.configure(
            "Fill.TCheckbutton",
            background="#d9d9d9",
            foreground="#000000",
            font=BUTTON_FONT,
            relief="raised",
            padding=BUTTON_PAD,
        )
        style.map(
            "Fill.TCheckbutton",
            background=[("active", "#c0c0c0"), ("selected", "#4a90e2")],
            relief=[("selected", "sunken"), ("!selected", "raised")],
            foreground=[("selected", "#ffffff"), ("!selected", "#000000")],
        )

    def _build_ui(self) -> None:
        """
        Assemble the main user interface.  A paned window divides the
        annotation canvas from the scrollable control panel.  The control
        panel is built up from a number of helper methods which populate
        ``self.ctrl_frame``.
        """
        self._build_styles()
        # Configure grid weights on the root frame so that the paned window
        # expands to fill the available space.
        for col, weight in COLUMN_WEIGHTS.items():
            self.root.columnconfigure(col, weight=weight)
        for row, weight in ROW_WEIGHTS.items():
            self.root.rowconfigure(row, weight=weight)

        # Create a paned window to separate the canvas and the control panel
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # Canvas for displaying the image and drawing annotations
        self.canvas = tk.Canvas(
            self.root,
            bg=CANVAS_BG,
            cursor=CANVAS_CURSOR,
            highlightthickness=0,
        )
        self.paned.add(self.canvas, weight=1)
        # Bind mouse events for drawing, panning and zooming
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<ButtonPress-3>", self._on_right_press)
        self.canvas.bind("<B3-Motion>", self._on_right_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_release)
        # Cross platform mousewheel bindings
        self.canvas.bind(
            "<MouseWheel>", lambda e: self._wrap_cmd(self._mousewheel_zoom)(e)
        )
        self.canvas.bind(
            "<Button-4>", lambda e: self._wrap_cmd(lambda _=e: self._set_zoom(1.1))(e)
        )
        self.canvas.bind(
            "<Button-5>", lambda e: self._wrap_cmd(lambda _=e: self._set_zoom(0.9))(e)
        )

        # Build the scrollable control panel
        self._build_control_panel()

        # Bind global scroll wheel events to decide whether to zoom or scroll
        self.root.bind_all("<MouseWheel>", self._on_global_wheel)
        self.root.bind_all("<Button-4>", self._on_global_wheel)  # Linux up
        self.root.bind_all("<Button-5>", self._on_global_wheel)  # Linux down

    def _build_control_panel(self) -> None:
        """Construct the control panel and populate it with UI elements."""
        ctrl_container = ttk.Frame(self.paned)
        self.paned.add(ctrl_container, weight=1)
        # Canvas for scrolling
        self.ctrl_canvas = tk.Canvas(
            ctrl_container,
            borderwidth=0,
            highlightthickness=0,
            background=BACKGROUND,
        )
        vsb = ttk.Scrollbar(
            ctrl_container, orient="vertical", command=self.ctrl_canvas.yview
        )
        hsb = ttk.Scrollbar(
            ctrl_container, orient="horizontal", command=self.ctrl_canvas.xview
        )
        self.ctrl_canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.ctrl_canvas.pack(side="left", fill="both", expand=True)
        # Frame inside the canvas to hold all controls
        self.ctrl_frame = ttk.Frame(self.ctrl_canvas, padding=(PAD_SMALL, PAD_SMALL))
        self.ctrl_frame_id = self.ctrl_canvas.create_window(
            (0, 0), window=self.ctrl_frame, anchor="nw"
        )
        # Update scrollregion whenever the size changes
        self.ctrl_frame.bind(
            "<Configure>",
            lambda e: self.ctrl_canvas.configure(
                scrollregion=self.ctrl_canvas.bbox("all")
            ),
        )
        # When entering/leaving the control frame, bind/unbind scroll wheel to scroll
        self.ctrl_frame.bind("<Enter>", lambda e: self._bind_control_scroll())
        self.ctrl_frame.bind("<Leave>", lambda e: self._unbind_control_scroll())

        # Build sections of the control panel in the order they should appear.
        # The inference panel is constructed inside the menu (top row) so it is
        # omitted here.  The search bar is placed above the main controls.
        self._build_menu()
        self._build_search_controls()
        self._build_navigation_controls()
        self._build_zoom_controls()
        self._build_metadata_controls()
        self._build_defect_controls()
        self._build_fill_toggle()
        self._build_pen_controls()
        self._build_export_controls()

    def _build_menu(self) -> None:
        """Settings menu and optional logo at the top of the control panel."""
        top_row = ttk.Frame(self.ctrl_frame)
        # Place the top row at the very top of the control panel
        top_row.grid(row=0, column=0, columnspan=3, sticky="ew")
        settings_btn = tk.Menubutton(top_row, text="⚙", relief="raised")
        settings_menu = tk.Menu(settings_btn, tearoff=0)
        settings_btn["menu"] = settings_menu
        settings_btn.grid(row=0, column=0, sticky="w", padx=PAD_SMALL, pady=PAD_SMALL)
        settings_menu.add_command(
            label="Input Folder…", command=self.choose_input_folder
        )
        settings_menu.add_command(
            label="Defects JSON…", command=self.choose_defects_config
        )
        settings_menu.add_command(
            label="Output Subdir…", command=self.choose_output_subdir
        )
        settings_menu.add_command(label="Load Model…", command=self.choose_model)
        # Logo positioned in the middle of the top row
        logo_path = os.path.join(os.path.dirname(__file__), "imgs", "AnnoMate.png")
        try:
            logo_img = Image.open(logo_path)
            logo_img = logo_img.resize((100, 100), Image.LANCZOS)
            self.logo_tk = ImageTk.PhotoImage(logo_img)
            logo_label = ttk.Label(top_row, image=self.logo_tk, background=BACKGROUND)
            logo_label.grid(
                row=0,
                column=1,
                sticky="e",
                padx=PAD_SMALL,
                pady=PAD_SMALL,
            )
        except Exception as ex:
            logger.warning(f"Failed to load logo: {ex}")
        # Place inference controls to the right of the logo
        # They will appear on the opposite side of the logo relative to the settings button
        self._build_inference_controls(parent=top_row, row=0, column=2)
        # Ensure the rightmost column expands if space allows
        top_row.columnconfigure(2, weight=1)

    def _build_navigation_controls(self) -> None:
        """Create navigation buttons for image traversal and editing."""
        nav_frame = ttk.LabelFrame(
            self.ctrl_frame,
            text="Main Controls",
            padding=(PAD_SMALL, PAD_SMALL),
        )
        nav_frame.grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=PAD_SMALL,
            pady=(0, PAD_SMALL),
        )
        actions: List[Tuple[str, Callable[[], None]]] = [
            ("◀ Prev", self.prev_image),
            ("Next ▶", self.next_image),
            ("Clear", self.clear_all),
            ("Undo", self.undo),
            ("Redo", self.redo),
            ("Draw", lambda: self.set_mode("draw")),
            ("Erase", lambda: self.set_mode("erase")),
            ("Save", self.save),
        ]
        for i, (txt, fn) in enumerate(actions):
            btn = ttk.Button(nav_frame, text=txt, command=self._wrap_cmd(fn))
            btn.grid(
                row=i // 2,
                column=i % 2,
                sticky="ew",
                padx=BUTTON_PAD[0],
                pady=BUTTON_PAD[1],
            )
        for c in range(2):
            nav_frame.columnconfigure(c, weight=1)

    def _build_zoom_controls(self) -> None:
        """Create a small set of buttons for zooming in and out."""
        zoom_frame = ttk.LabelFrame(
            self.ctrl_frame,
            text="Zoom",
            padding=(PAD_SMALL, PAD_SMALL),
        )
        zoom_frame.grid(
            row=3, column=0, sticky="ew", padx=PAD_SMALL, pady=(0, PAD_SMALL)
        )
        for symbol, factor in [("−", 0.9), ("+", 1.1)]:
            ttk.Button(
                zoom_frame,
                text=symbol,
                command=self._wrap_cmd(lambda f=factor: self._set_zoom(f)),
            ).pack(side="left", padx=BUTTON_PAD[0], pady=BUTTON_PAD[1])

    def _build_metadata_controls(self) -> None:
        """Create fields for notes, inspector, tray and filename information."""
        meta_frame = ttk.LabelFrame(
            self.ctrl_frame,
            text="Information",
            padding=(PAD_SMALL, PAD_SMALL),
        )
        meta_frame.grid(
            row=4, column=0, sticky="ew", padx=PAD_SMALL, pady=(0, PAD_SMALL)
        )
        # Notes
        ttk.Label(meta_frame, text="Note:").grid(
            row=0, column=0, sticky="nw", padx=PAD_SMALL
        )
        self.note_text = tk.Text(meta_frame, wrap="word", font=ENTRY_FONT, height=3)
        self.note_text.grid(row=0, column=1, sticky="ew", padx=PAD_SMALL)
        self.note_text.bind("<KeyRelease>", lambda e: self._adjust_note_height())
        # Inspector
        ttk.Label(meta_frame, text="Inspector:").grid(
            row=1, column=0, sticky="w", padx=PAD_SMALL
        )
        self.inspector_var = tk.StringVar(master=self.root)
        self.inspector_entry = ttk.Entry(
            meta_frame, textvariable=self.inspector_var, font=ENTRY_FONT
        )
        self.inspector_entry.grid(row=1, column=1, sticky="ew", padx=PAD_SMALL, pady=2)
        # Tray
        ttk.Label(meta_frame, text="Tray:").grid(
            row=2, column=0, sticky="w", padx=PAD_SMALL
        )
        self.tray_var = tk.StringVar(master=self.root)
        ttk.Entry(
            meta_frame, textvariable=self.tray_var, state="readonly", font=ENTRY_FONT
        ).grid(row=2, column=1, sticky="ew", padx=PAD_SMALL, pady=2)
        # Filename
        ttk.Label(meta_frame, text="Filename:").grid(
            row=3, column=0, sticky="w", padx=PAD_SMALL
        )
        # Do not reassign `self.filename_var` here.  This StringVar is
        # created in `__init__` and shared between the search bar and
        # metadata panel.  Reassigning it here would break the
        # connection to the search display.  Instead, reuse the existing
        # instance.
        ttk.Entry(
            meta_frame,
            textvariable=self.filename_var,
            state="readonly",
            font=ENTRY_FONT,
        ).grid(row=3, column=1, sticky="ew", padx=PAD_SMALL, pady=2)
        # Last saved
        ttk.Label(meta_frame, text="Last saved:").grid(
            row=4, column=0, sticky="w", padx=PAD_SMALL
        )
        self.save_ts_var = tk.StringVar(master=self.root, value="never")
        ttk.Label(meta_frame, textvariable=self.save_ts_var).grid(
            row=4, column=1, sticky="w", padx=PAD_SMALL, pady=2
        )
        meta_frame.columnconfigure(1, weight=1)

    def _build_defect_controls(self) -> None:
        """Create a list of checkboxes, one per defect category."""
        def_frame = ttk.LabelFrame(
            self.ctrl_frame,
            text="Categories",
            padding=(PAD_SMALL, PAD_SMALL),
        )
        def_frame.grid(
            row=5, column=0, sticky="ew", padx=PAD_SMALL, pady=(0, PAD_SMALL)
        )
        self.defect_vars: Dict[str, tk.BooleanVar] = {}
        for i, d in enumerate(self.defects):
            var = tk.BooleanVar(master=self.root)
            cb = ttk.Checkbutton(
                def_frame, text=d, variable=var, command=self._update_pen_labels
            )
            cb.grid(row=i, column=0, sticky="w", padx=PAD_SMALL)
            # When a defect is toggled off, remove its marks
            var.trace_add(
                "write", lambda *args, defect=d: self._on_defect_toggle(defect)
            )
            self.defect_vars[d] = var

    def _build_fill_toggle(self) -> None:
        """Create a toggle for enabling flood‑fill mode on the current pen."""
        fill_frame = ttk.LabelFrame(
            self.ctrl_frame,
            text="Fill Mode",
            padding=(PAD_SMALL, PAD_SMALL),
        )
        fill_frame.grid(
            row=6, column=0, sticky="ew", padx=PAD_SMALL, pady=(0, PAD_SMALL)
        )
        self.fill_var = tk.BooleanVar(master=self.root)
        ttk.Checkbutton(
            fill_frame,
            text="Pen Fill Mode",
            variable=self.fill_var,
            style="Fill.TCheckbutton",
        ).pack(padx=PAD_SMALL, pady=PAD_SMALL)

    def _build_pen_controls(self) -> None:
        """Create pen selection and labelling controls."""
        self.pen_frame = ttk.LabelFrame(
            self.ctrl_frame,
            text="Pen Labels",
            padding=(PAD_SMALL, PAD_SMALL),
        )
        self.pen_frame.grid(
            row=7, column=0, sticky="ew", padx=PAD_SMALL, pady=(0, PAD_SMALL)
        )
        self.pen_vars: Dict[str, tk.StringVar] = {}
        self.pen_widgets: Dict[str, Tuple[ttk.Label, ttk.Combobox, ttk.Button]] = {}
        pen_order: List[Tuple[str, str]] = [
            ("pen1", "Red"),
            ("pen2", "Green"),
            ("pen3", "Blue"),
            ("pen4", "Orange"),
            ("pen5", "Purple"),
            ("pen6", "Cyan"),
        ]
        for i, (key, colour_name) in enumerate(pen_order):
            var = tk.StringVar(master=self.root)
            label = ttk.Label(self.pen_frame, text=f"{key.upper()} label:")
            combo = ttk.Combobox(
                self.pen_frame,
                textvariable=var,
                state="readonly",
                values=self.defects,
                font=ENTRY_FONT,
                width=25,
            )
            btn = ttk.Button(
                self.pen_frame,
                text=f"Use {key.upper()} ({colour_name})",
                command=self._wrap_cmd(lambda k=key: self.select_pen(k)),
            )
            self.pen_vars[key] = var
            self.pen_widgets[key] = (label, combo, btn)
        # Initially hide all pen widgets; they are shown for selected defects
        self._refresh_pen_widgets()

    def _build_search_controls(self) -> None:
        """Create a search entry for jumping to images by filename."""
        search_frame = ttk.Frame(self.ctrl_frame)
        search_frame.grid(
            row=2, column=0, sticky="ew", padx=PAD_SMALL, pady=(0, PAD_SMALL)
        )
        ttk.Label(search_frame, text="Search Image:").pack(side="left")
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20)
        search_entry.pack(side="left", padx=(PAD_SMALL, 0))
        search_entry.bind("<Return>", self._on_search_submit)
        ttk.Label(search_frame, text="Current Image:").pack(
            side="left", padx=(PAD_MEDIUM, 0)
        )
        ttk.Label(search_frame, textvariable=self.filename_var).pack(
            side="left", padx=(PAD_SMALL, 0)
        )
        search_frame.columnconfigure(0, weight=1)

    def _build_inference_controls(
        self,
        parent: Optional[tk.Widget] = None,
        row: Optional[int] = None,
        column: Optional[int] = None,
    ) -> None:
        """
        Display model prediction and confidence values.

        Parameters
        ----------
        parent: tk.Widget, optional
            The widget into which the inference controls should be
            gridded.  If ``None``, defaults to the main control frame.
        row: int, optional
            Grid row for the inference frame.  If ``None`` and the
            parent is the control frame, a default row is chosen; if
            provided, it overrides the default.
        column: int, optional
            Grid column for the inference frame.  Only used when
            specifying a parent.  Defaults to ``0`` when unspecified.
        """
        # Determine the parent container
        tgt = parent if parent is not None else self.ctrl_frame
        # Default row when placed in the control panel; allow override
        r = row if row is not None else 8
        c = column if column is not None else 0
        inf_frame = ttk.LabelFrame(
            tgt,
            text="Inference",
            padding=(PAD_SMALL, PAD_SMALL),
        )
        inf_frame.grid(
            row=r,
            column=c,
            sticky="ew",
            padx=PAD_SMALL,
            pady=(0, PAD_SMALL),
        )
        # Initialise variables for predicted class and confidence
        self.infer_result_var = tk.StringVar(master=self.root, value="Result: N/A")
        self.infer_conf_var = tk.StringVar(master=self.root, value="Confidence: N/A")
        ttk.Label(inf_frame, textvariable=self.infer_result_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(inf_frame, textvariable=self.infer_conf_var).grid(
            row=1, column=0, sticky="w"
        )
        inf_frame.columnconfigure(0, weight=1)

    def _build_export_controls(self) -> None:
        """Create buttons for exporting metadata to Excel or CSV."""
        export_frame = ttk.LabelFrame(
            self.ctrl_frame,
            text="Export Information",
            padding=(PAD_SMALL, PAD_SMALL),
        )
        export_frame.grid(
            row=8, column=0, sticky="ew", padx=PAD_SMALL, pady=(0, PAD_SMALL)
        )
        ttk.Button(
            export_frame,
            text="Export as XLSX",
            command=self._wrap_cmd(self.export_excel),
        ).pack(side="left", padx=PAD_SMALL)
        ttk.Button(
            export_frame,
            text="Export as CSV",
            command=self._wrap_cmd(self.export_csv),
        ).pack(side="left", padx=PAD_SMALL)

    # ---------------------------------------------------------------------
    # Internal helper methods
    # ---------------------------------------------------------------------

    def _bind_control_scroll(self) -> None:
        """Bind scroll events to the control panel when the cursor enters."""
        self.ctrl_canvas.bind_all("<MouseWheel>", self._on_control_panel_wheel)
        self.ctrl_canvas.bind_all("<Button-4>", self._on_control_panel_wheel)
        self.ctrl_canvas.bind_all("<Button-5>", self._on_control_panel_wheel)

    def _unbind_control_scroll(self) -> None:
        """Unbind scroll events when the cursor leaves the control panel."""
        self.ctrl_canvas.unbind_all("<MouseWheel>")
        self.ctrl_canvas.unbind_all("<Button-4>")
        self.ctrl_canvas.unbind_all("<Button-5>")

    def _preload_predictions(self) -> None:
        """
        Precompute model predictions for each image in the dataset.  This
        method displays a progress bar and populates ``self.pred_cache`` with
        tuples of (orig, heatmap, overlay, summary, pred, prob).  Any
        failures during inference are logged and stored as ``None``.
        """
        total = len(self.image_paths)
        for i, path in enumerate(self.image_paths):
            try:
                orig, heat, ov, summary, pred, prob = run_inference(
                    str(self.model_path), path
                )
                self.pred_cache[path] = (orig, heat, ov, summary, pred, prob)
            except Exception as ex:
                logger.error(f"Inference failed for {path}: {ex}")
                self.pred_cache[path] = None
            # Update progress bar (percentage)
            self._progress_var.set((i + 1) / total * 100)
            self.root.update_idletasks()

    def _wrap_cmd(
        self, fn: Callable[[Optional[tk.Event]], None]
    ) -> Callable[[Optional[tk.Event]], None]:
        """
        Wrap a callback so that enabling fill mode does not persist across
        operations that should disable it (e.g. zooming, navigation).

        Parameters
        ----------
        fn: Callable
            The function to wrap.  It must accept zero or one positional
            argument (the optional event).

        Returns
        -------
        Callable
            A new function that first disables fill mode then calls ``fn``.
        """

        def wrapped(*args, **kwargs):
            # Whenever a button is clicked, disable fill mode
            self.fill_var.set(False)
            return fn(*args, **kwargs)

        return wrapped

    def _on_defect_toggle(self, defect: str) -> None:
        """
        Callback invoked when a defect checkbox is toggled.  If the defect
        has been unchecked, remove all strokes painted with its colour.
        """
        if not self.defect_vars[defect].get():
            self._remove_defect_marks(defect)
            self._show_image()
            if self.on_image_change:
                self.on_image_change(self.image_paths[self.idx], self.layer)

    def _remove_defect_marks(self, defect: str) -> None:
        """
        Remove all pixels from the overlay corresponding to a particular defect.
        Colour comparison is tolerant to small differences defined by
        ``COLOR_TOLERANCE``.
        """
        # Find the colour associated with this defect
        pen_key = DEFECT_TO_PEN.get(defect)
        if not pen_key:
            return
        target_colour = PEN_COLORS.get(pen_key)
        if not target_colour:
            return
        pr, pg, pb = target_colour[:3]
        w, h = self.base.size
        layer_pixels = self.layer.load()
        for x in range(w):
            for y in range(h):
                r, g, b, a = layer_pixels[x, y]
                if a == 0:
                    continue
                if (
                    abs(r - pr) <= COLOR_TOLERANCE
                    and abs(g - pg) <= COLOR_TOLERANCE
                    and abs(b - pb) <= COLOR_TOLERANCE
                ):
                    layer_pixels[x, y] = (0, 0, 0, 0)

    def _on_global_wheel(self, event: tk.Event) -> None:
        """
        Determine whether the mouse wheel should zoom the image or scroll the
        control panel based on the cursor position.  If the cursor is over
        the canvas we zoom, otherwise we scroll the controls.
        """
        # Pointer position in screen coordinates
        x, y = event.x_root, event.y_root
        # Canvas bounding box in screen coordinates
        ix1 = self.canvas.winfo_rootx()
        iy1 = self.canvas.winfo_rooty()
        ix2 = ix1 + self.canvas.winfo_width()
        iy2 = iy1 + self.canvas.winfo_height()
        if ix1 <= x <= ix2 and iy1 <= y <= iy2:
            # Inside the image canvas: zoom
            if hasattr(event, "delta") and event.delta:
                factor = 1.1 if event.delta > 0 else 0.9
            else:
                # Linux: Button-4 and Button-5
                factor = 1.1 if event.num == 4 else 0.9
            self._set_zoom(factor)
        else:
            # Over the control panel: scroll
            if hasattr(event, "delta") and event.delta:
                units = int(-1 * (event.delta / 120))
            else:
                units = -1 if event.num == 4 else 1
            self.ctrl_canvas.yview_scroll(units, "units")

    def _on_control_panel_wheel(self, event: tk.Event) -> None:
        """Scroll the control panel when the mouse wheel is used over it."""
        if hasattr(event, "delta") and event.delta:
            units = int(-1 * (event.delta / 120))
        else:
            units = -1 if event.num == 4 else 1
        self.ctrl_canvas.yview_scroll(units, "units")

    def _update_pen_labels(self, *args) -> None:
        """
        Assign each selected defect to its dedicated pen.  Pens are fixed to
        specific defects irrespective of the selection order.
        """
        # Clear all pen variables
        for k in self.pen_vars.keys():
            self.pen_vars[k].set("")
        # Assign selected defects to their pens
        for defect, var in self.defect_vars.items():
            if var.get():
                pen_key = DEFECT_TO_PEN.get(defect)
                if pen_key in self.pen_vars:
                    self.pen_vars[pen_key].set(defect)
        # Show/hide pen widgets accordingly
        self._refresh_pen_widgets()

    def _refresh_pen_widgets(self) -> None:
        """
        Update the visibility of pen widgets.  Only pens corresponding to
        selected defect categories are shown.  Widgets are arranged in
        successive rows.
        """
        # Hide all pen widgets
        for widgets in self.pen_widgets.values():
            for w in widgets:
                w.grid_remove()
        # Determine which pens should be visible
        visible_keys: List[str] = []
        for defect, var in self.defect_vars.items():
            if var.get():
                pen_key = DEFECT_TO_PEN.get(defect)
                if pen_key:
                    visible_keys.append(pen_key)
        visible_keys = sorted(set(visible_keys))
        # Arrange visible pen widgets in the pen_frame
        for i, key in enumerate(visible_keys):
            label, combo, btn = self.pen_widgets[key]
            label.grid(row=2 * i, column=0, sticky="w", padx=PAD_SMALL)
            combo.grid(row=2 * i, column=1, sticky="ew", padx=PAD_SMALL)
            btn.grid(
                row=2 * i + 1,
                column=0,
                columnspan=2,
                sticky="w",
                padx=PAD_SMALL,
                pady=(0, PAD_SMALL),
            )
        self.pen_frame.columnconfigure(1, weight=1)

    def _adjust_note_height(self, event: Optional[tk.Event] = None) -> None:
        """Grow the notes text widget to fit its content up to a minimum height."""
        text_widget = self.note_text
        line_count = int(text_widget.index("end-1c").split(".")[0])
        text_widget.configure(height=max(3, line_count))

    def select_pen(self, which: str) -> None:
        """
        Select the current pen.  Update the pen colour, width and cursor
        accordingly and revert to drawing mode.
        """
        self.current_pen = which
        self.pen_color = PEN_COLORS.get(which, (255, 0, 0, 255))
        self.pen_width = 8
        self._tk_pen_color = "#%02x%02x%02x" % self.pen_color[:3]
        self.mode = "draw"
        self.canvas.config(cursor="pencil")

    # ---------------------------------------------------------------------
    # Image loading and display
    # ---------------------------------------------------------------------

    def _load_image(self) -> None:
        """
        Load the current image, any existing combined mask and associated
        metadata.  Resets state such as history and pen assignments.
        """
        full_path = self.image_paths[self.idx]
        logger.info(f"Loading image: {full_path}")
        self.base = Image.open(full_path).convert("RGB")
        # Update tray and filename display
        folder = os.path.basename(os.path.dirname(full_path))
        file_name = os.path.basename(full_path)
        base_name, _ = os.path.splitext(file_name)
        self.tray_var.set(folder.split("_")[0])
        self.filename_var.set(f"{folder}/{file_name}")
        # Clear transient state (notes, defect selections, pen labels)
        self.note_text.delete("1.0", "end")
        for var in self.defect_vars.values():
            var.set(False)
        for var in self.pen_vars.values():
            var.set("")
        # Ensure masks directory exists
        masks_folder = os.path.join(self.output_dir, "masks")
        os.makedirs(masks_folder, exist_ok=True)
        # Try to load the most recently saved combined mask for this image
        pattern = os.path.join(masks_folder, f"*-COMBINED-{base_name}.png")
        mask_files = glob.glob(pattern)
        if mask_files:
            latest_mask = max(mask_files, key=os.path.getmtime)
            lm = Image.open(latest_mask).convert("RGBA")
            if lm.size != self.base.size:
                lm = lm.resize(self.base.size, Image.LANCZOS)
            self.layer = lm
        else:
            self.layer = Image.new("RGBA", self.base.size, (0, 0, 0, 0))
            # Save an initial placeholder combined mask to disk
            self._save_current()
        # Drawer for drawing onto the overlay
        self.drawer = ImageDraw.Draw(self.layer)
        # Reset undo/redo history and panning state
        self.history = [self.layer.copy()]
        self.redo_stack.clear()
        self.pan_x = self.pan_y = 0
        self.zoom = 1.0
        self.drawing = False
        # Load metadata for this image (note, inspector, tray, defects, pen labels, last saved)
        key = os.path.basename(full_path)
        meta = self._load_meta().get(key, {})
        if "inspector" in meta:
            self.inspector_var.set(meta["inspector"])
        self.note_text.delete("1.0", "end")
        self.note_text.insert("1.0", meta.get("note", ""))
        if meta.get("tray"):
            self.tray_var.set(meta["tray"])
        for defect, var in self.defect_vars.items():
            var.set(defect in meta.get("defects", []))
        self._update_pen_labels()
        for pen_key, lab in meta.get("pen_labels", {}).items():
            if pen_key in self.pen_vars:
                self.pen_vars[pen_key].set(lab)
        self.save_ts_var.set(meta.get("last_saved", "never"))
        self._unsaved_clear = False

    def _show_image(self) -> None:
        """Compose the base image and overlay and display the result."""
        comp = Image.alpha_composite(self.base.convert("RGBA"), self.layer)
        w, h = comp.size
        disp = comp.resize((int(w * self.zoom), int(h * self.zoom)), Image.LANCZOS)
        self.tkimg = ImageTk.PhotoImage(disp)
        self.canvas.delete("IMG")
        self.canvas.create_image(
            self.pan_x,
            self.pan_y,
            anchor="nw",
            image=self.tkimg,
            tags="IMG",
        )
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def _fit_image_to_canvas(self) -> None:
        """Compute an initial zoom factor to fit the image within the canvas."""
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw > 1 and ch > 1:
            bw, bh = self.base.size
            self.zoom = min(cw / bw, ch / bh, 1.0)

    # ---------------------------------------------------------------------
    # Drawing and panning event handlers
    # ---------------------------------------------------------------------

    def _on_left_press(self, event: tk.Event) -> None:
        """
        Handle the left mouse button press.  If fill mode is active perform a
        flood fill at the clicked position; otherwise start a new stroke.
        """
        self._current_stroke_ids = []
        cx = self.canvas.canvasx(event.x) - self.pan_x
        cy = self.canvas.canvasy(event.y) - self.pan_y
        ix, iy = int(cx / self.zoom), int(cy / self.zoom)
        if self.fill_var.get():
            # Flood fill on the overlay using the current pen colour
            ImageDraw.floodfill(self.layer, (ix, iy), self.pen_color)
            self._record_history()
            self._show_image()
            if self.on_image_change:
                self.on_image_change(self.image_paths[self.idx], self.layer)
            return
        # Begin a freehand stroke
        self.drawing = True
        self.stroke_start_x, self.stroke_start_y = ix, iy
        self.last_x, self.last_y = ix, iy
        self._show_image()
        if self.on_image_change:
            self.on_image_change(self.image_paths[self.idx], self.layer)

    def _on_left_move(self, event: tk.Event) -> None:
        """Draw a line segment as the mouse moves while the left button is held."""
        if self.fill_var.get() or not self.drawing:
            return
        cx = self.canvas.canvasx(event.x) - self.pan_x
        cy = self.canvas.canvasy(event.y) - self.pan_y
        ix, iy = int(cx / self.zoom), int(cy / self.zoom)
        x1 = self.last_x * self.zoom + self.pan_x
        y1 = self.last_y * self.zoom + self.pan_y
        x2 = ix * self.zoom + self.pan_x
        y2 = iy * self.zoom + self.pan_y
        sid = self.canvas.create_line(
            x1,
            y1,
            x2,
            y2,
            fill=self._tk_pen_color,
            width=self.pen_width * self.zoom,
            capstyle="round",
            smooth=True,
        )
        self._current_stroke_ids.append(sid)
        self.last_x, self.last_y = ix, iy

    def _on_left_release(self, event: tk.Event) -> None:
        """
        Finish the freehand stroke when the left mouse button is released.
        Draw the stroke onto the overlay image and record it for undo.
        """
        if self.drawing and self.mode == "draw" and not self.fill_var.get():
            drawer = ImageDraw.Draw(self.layer)
            # For each temporary line segment drawn on the canvas, map its
            # coordinates back to image space and draw onto the overlay
            for sid in self._current_stroke_ids:
                coords = self.canvas.coords(sid)
                for i in range(0, len(coords) - 2, 2):
                    ix1 = (coords[i] - self.pan_x) / self.zoom
                    iy1 = (coords[i + 1] - self.pan_y) / self.zoom
                    ix2 = (coords[i + 2] - self.pan_x) / self.zoom
                    iy2 = (coords[i + 3] - self.pan_y) / self.zoom
                    drawer.line(
                        [(ix1, iy1), (ix2, iy2)],
                        fill=self.pen_color,
                        width=self.pen_width,
                    )
            # Connect the last point back to the starting point (ensures a
            # single continuous stroke)
            drawer.line(
                [
                    (self.last_x, self.last_y),
                    (self.stroke_start_x, self.stroke_start_y),
                ],
                fill=self.pen_color,
                width=self.pen_width,
            )
            # Remove temporary strokes from the canvas
            for sid in self._current_stroke_ids:
                self.canvas.delete(sid)
            self._current_stroke_ids = []
            self._record_history()
            self._show_image()
            if self.on_image_change:
                self.on_image_change(self.image_paths[self.idx], self.layer)
        self.drawing = False

    def _on_right_press(self, event: tk.Event) -> None:
        """Start panning the view on right mouse button press."""
        self._prx, self._pry = event.x, event.y

    def _on_right_move(self, event: tk.Event) -> None:
        """Move the view while the right mouse button is held."""
        dx, dy = event.x - self._prx, event.y - self._pry
        self._prx, self._pry = event.x, event.y
        self.pan_x += dx
        self.pan_y += dy
        self._show_image()
        if self.on_image_change:
            self.on_image_change(self.image_paths[self.idx], self.layer)

    def _on_right_release(self, event: tk.Event) -> None:
        """End panning on right mouse button release."""
        self._prx = self._pry = None

    def _mousewheel_zoom(self, event: tk.Event) -> None:
        """Zoom the image smoothly when the mouse wheel is used over the canvas."""
        self.zoom *= 1.1 if event.delta > 0 else 0.9
        if self._zoom_after_id:
            self.canvas.after_cancel(self._zoom_after_id)
        self._zoom_after_id = self.canvas.after_idle(self._apply_zoom)

    def _apply_zoom(self) -> None:
        """Redraw the image after the zoom factor has changed."""
        self._zoom_after_id = None
        self._show_image()

    def set_mode(self, mode: str) -> None:
        """Switch between draw and erase modes."""
        self.mode = mode
        self.canvas.config(cursor="pencil" if mode == "draw" else "circle")

    def _set_zoom(self, factor: float) -> None:
        """Multiply the zoom factor by ``factor`` and constrain it to a sensible range."""
        self.zoom = max(0.1, min(self.zoom * factor, 10.0))
        if self.on_image_change:
            self.on_image_change(self.image_paths[self.idx], self.layer)
        self._show_image()

    # ---------------------------------------------------------------------
    # Undo/redo and navigation
    # ---------------------------------------------------------------------

    def _record_history(self) -> None:
        """Append the current overlay to the history, pruning old entries."""
        self.history.append(self.layer.copy())
        if len(self.history) > 50:
            self.history.pop(0)
        self.redo_stack.clear()

    def clear_all(self) -> None:
        """
        Clear all annotations on the current image and reset metadata fields.
        """
        self.layer = Image.new("RGBA", self.base.size, (0, 0, 0, 0))
        self.drawer = ImageDraw.Draw(self.layer)
        self.history = [self.layer.copy()]
        self.redo_stack.clear()
        self.note_text.delete("1.0", "end")
        self.inspector_var.set("")
        for var in self.defect_vars.values():
            var.set(False)
        for var in self.pen_vars.values():
            var.set("")
        self._show_image()
        if self.on_image_change:
            self.on_image_change(self.image_paths[self.idx], self.layer)
        self._unsaved_clear = True

    def undo(self) -> None:
        """Undo the most recent drawing action."""
        if len(self.history) > 1:
            self.redo_stack.append(self.history.pop())
            self.layer = self.history[-1].copy()
            self.drawer = ImageDraw.Draw(self.layer)
            self._show_image()
            if self.on_image_change:
                self.on_image_change(self.image_paths[self.idx], self.layer)

    def redo(self) -> None:
        """Redo the most recently undone drawing action."""
        if self.redo_stack:
            img = self.redo_stack.pop()
            self.history.append(img.copy())
            self.layer = img.copy()
            self.drawer = ImageDraw.Draw(self.layer)
            self._show_image()
            if self.on_image_change:
                self.on_image_change(self.image_paths[self.idx], self.layer)

    def prev_image(self) -> None:
        """
        Navigate to the previous image in the folder, saving the current
        annotations if necessary.
        """
        if not self._unsaved_clear and not hasattr(self, "_combined_ts"):
            self._save_current()
        else:
            self._unsaved_clear = False
        self.idx = (self.idx - 1) % len(self.image_paths)
        self._load_image()
        self._fit_image_to_canvas()
        self._show_image()
        if self.on_image_change:
            self.on_image_change(self.image_paths[self.idx], self.layer)

    def next_image(self) -> None:
        """
        Navigate to the next image in the folder, saving the current
        annotations if necessary.
        """
        if not self._unsaved_clear and not hasattr(self, "_combined_ts"):
            self._save_current()
        else:
            self._unsaved_clear = False
        self.idx = (self.idx + 1) % len(self.image_paths)
        self._load_image()
        self._fit_image_to_canvas()
        self._show_image()
        if self.on_image_change:
            self.on_image_change(self.image_paths[self.idx], self.layer)

    def _on_close(self) -> None:
        """Handle window closure by saving the current overlay if needed."""
        if not self._unsaved_clear:
            self._save_current()
        self.root.destroy()

    def save(self) -> None:
        """
        Save individual masks for each selected defect and a combined overlay.
        Metadata is updated and written to disk.  After saving a message
        box informs the user of the saved defects.
        """
        logger.info(f"Saving annotations for image index {self.idx}")
        now = datetime.datetime.now()
        file_ts = now.strftime(TS_FILENAME_FMT)
        disp_ts = now.strftime(TS_DISPLAY_FMT)
        inspector = self.inspector_var.get().strip() or "unknown"
        tray = self.tray_var.get().strip() or ""
        input_folder_name = os.path.basename(self.input_dir.rstrip(os.sep))
        base_name = os.path.splitext(os.path.basename(self.image_paths[self.idx]))[0]
        selected_defects: List[str] = [
            d for d, v in self.defect_vars.items() if v.get()
        ]
        # Build a mapping from pen key to colour (RGB only)
        pen_to_color_rgb: Dict[str, Tuple[int, int, int]] = {
            k: v[:3] for k, v in PEN_COLORS.items()
        }
        # Save individual defect masks
        for defect in selected_defects:
            out_dir = os.path.join(self.output_dir, defect)
            os.makedirs(out_dir, exist_ok=True)
            # Remove any previous masks for this image/defect
            pattern = os.path.join(
                out_dir, f"{input_folder_name}-*-{defect}-{base_name}.png"
            )
            for fp in glob.glob(pattern):
                try:
                    os.remove(fp)
                except OSError:
                    pass
            # Determine which pens correspond to this defect
            matching_colors: List[Tuple[int, int, int]] = []
            for pen_key, label_text in self.pen_vars.items():
                if label_text.get().strip().lower() == defect.strip().lower():
                    matching_colors.append(pen_to_color_rgb[pen_key])
            # Build a mask containing only pixels matching the defect colours
            mask_rgba = Image.new("RGBA", self.base.size, (0, 0, 0, 0))
            src = self.layer.load()
            dst = mask_rgba.load()
            w, h = self.base.size
            for x in range(w):
                for y in range(h):
                    r, g, b, a = src[x, y]
                    if a == 0:
                        continue
                    for pr, pg, pb in matching_colors:
                        if (
                            abs(r - pr) <= COLOR_TOLERANCE
                            and abs(g - pg) <= COLOR_TOLERANCE
                            and abs(b - pb) <= COLOR_TOLERANCE
                        ):
                            dst[x, y] = (pr, pg, pb, a)
                            break
            # Flatten the mask and draw the defect label in the corner
            rgb_mask = Image.new("RGB", self.base.size, (0, 0, 0))
            rgb_mask.paste(mask_rgba, mask=mask_rgba.split()[3])
            draw_legend = ImageDraw.Draw(rgb_mask)
            font = ImageFont.load_default()
            bbox = draw_legend.textbbox((0, 0), defect, font=font)
            text_h = bbox[3] - bbox[1]
            draw_legend.text(
                (5, self.base.size[1] - PAD_SMALL - text_h),
                defect,
                fill=(255, 255, 255),
                font=font,
            )
            out_filename = (
                f"{input_folder_name}-{inspector}-{file_ts}-{defect}-{base_name}.png"
            )
            out_path = os.path.join(out_dir, out_filename)
            rgb_mask.save(out_path)
        # Remove masks for deselected defects
        all_defects = list(self.defect_vars.keys())
        for defect in all_defects:
            if defect not in selected_defects:
                out_dir = os.path.join(self.output_dir, defect)
                pattern = os.path.join(
                    out_dir, f"{input_folder_name}-*-{defect}-{base_name}.png"
                )
                for fp in glob.glob(pattern):
                    try:
                        os.remove(fp)
                    except OSError:
                        pass
        # Combined mask overwrite logic
        combined_dir = os.path.join(self.output_dir, "masks")
        os.makedirs(combined_dir, exist_ok=True)
        short_name = f"{input_folder_name}-COMBINED-{base_name}.png"
        short_path = os.path.join(combined_dir, short_name)
        if not hasattr(self, "_combined_ts"):
            now = datetime.datetime.now()
            self._combined_ts = now.strftime(TS_FILENAME_FMT)
        all_defects_str = "_".join(selected_defects) or "good"
        long_name = f"{input_folder_name}-{inspector}-{self._combined_ts}-{all_defects_str}-COMBINED-{base_name}.png"
        long_path = os.path.join(combined_dir, long_name)
        combined_pattern = os.path.join(
            combined_dir, f"{input_folder_name}*-COMBINED-{base_name}.png"
        )
        for fp in glob.glob(combined_pattern):
            try:
                os.remove(fp)
            except OSError:
                pass
        short_pattern = os.path.join(
            combined_dir, f"{input_folder_name}-COMBINED-{base_name}.png"
        )
        for placeholder in glob.glob(short_pattern):
            try:
                os.remove(placeholder)
            except OSError:
                logger.warning(f"Failed to remove placeholder mask: {placeholder}")
        self.layer.save(long_path)
        # Update metadata
        meta_path = self._meta_path()
        meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
        key = os.path.basename(self.image_paths[self.idx])
        entry = meta.get(key, {})
        entry.update(
            {
                "note": self.note_text.get("1.0", "end-1c"),
                "inspector": self.inspector_var.get(),
                "tray": self.tray_var.get(),
                "defects": selected_defects,
                "pen_labels": {k: v.get() for k, v in self.pen_vars.items()},
                "last_saved": disp_ts,
                "model": self.model_status,
                "confidence": self.model_confidence,
            }
        )
        meta[key] = entry
        json.dump(meta, open(meta_path, "w"), indent=2)
        self.save_ts_var.set(disp_ts)
        export_list = selected_defects or ["good"]
        messagebox.showinfo(
            "Saved", f"Masks exported to: {', '.join(export_list)}"
        )
        logger.info(f"Masks exported to: {', '.join(export_list)}")

    def _save_current(self) -> None:
        """
        Save the current overlay as a placeholder combined mask.  This is
        invoked automatically when navigating away from an image without
        explicitly calling :meth:`save` to ensure that nothing is lost.
        """
        logger.debug(f"Auto-saving placeholder mask for index {self.idx}")
        name = os.path.splitext(os.path.basename(self.image_paths[self.idx]))[0]
        input_folder_name = os.path.basename(self.input_dir.rstrip(os.sep))
        mask_dir = os.path.join(self.output_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        mask_name = f"{input_folder_name}-COMBINED-{name}.png"
        mp = os.path.join(mask_dir, mask_name)
        self.layer.save(mp)
        meta_path = self._meta_path()
        meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
        key = os.path.basename(self.image_paths[self.idx])
        entry = meta.get(key, {})
        entry.update(
            {
                "note": self.note_text.get("1.0", "end-1c"),
                "inspector": self.inspector_var.get(),
                "tray": self.tray_var.get(),
                "defects": [d for d, v in self.defect_vars.items() if v.get()],
                "pen_labels": {k: v.get() for k, v in self.pen_vars.items()},
                "last_saved": entry.get("last_saved", "never"),
            }
        )
        meta[key] = entry
        json.dump(meta, open(meta_path, "w"), indent=2)

    def export_excel(self) -> None:
        """
        Export metadata across all images to an Excel workbook.  Each row
        contains filename, tray/directory, inspector name, inspector
        prediction, model prediction, defects and notes.
        """
        logger.info("Exporting metadata to Excel")
        try:
            from openpyxl import Workbook
        except ImportError:
            messagebox.showerror(
                "Export Excel",
                "openpyxl is not installed. Install it to enable Excel export.",
            )
            return
        base_name = os.path.basename(self.input_dir.rstrip(os.sep))
        excel_path = os.path.join(self.output_dir, f"{base_name}.xlsx")
        wb = Workbook()
        ws = wb.active
        ws.title = "Tray Information"
        headers = [
            "Filename",
            "Tray/Directory",
            "Inspector",
            "Inspector Prediction",
            "Model Prediction",
            "Defect(s)",
            "Notes",
        ]
        ws.append(headers)
        meta = self._load_meta()
        for fp in self.image_paths:
            bf = os.path.basename(fp)
            name, _ = os.path.splitext(bf)
            e = meta.get(bf, {})
            defects = e.get("defects", [])
            insp = e.get("inspector", "").strip()
            tray = e.get("tray", "").strip()
            note = e.get("note", "").strip()
            last_saved = e.get("last_saved", "").strip()
            model_pred = e.get("model", "N/A").strip()
            # Determine inspector's accept/reject status
            if not defects and last_saved in ("never", ""):
                insp_pred = "Unlabeled"
            elif not defects:
                insp_pred = "Accept"
            else:
                insp_pred = "Reject"
            ws.append(
                [
                    name,
                    tray,
                    insp,
                    insp_pred,
                    model_pred,
                    ", ".join(defects),
                    note,
                ]
            )
        # Adjust column widths
        for col in ws.columns:
            max_len = max((len(str(c.value)) for c in col), default=0)
            ws.column_dimensions[col[0].column_letter].width = max_len + 2
        wb.save(excel_path)
        messagebox.showinfo("Export Excel", f"Excel saved to:\n{excel_path}")

    def export_csv(self) -> None:
        """
        Export metadata across all images to a CSV file.  Each row
        contains filename, tray/directory, inspector name, accept/reject
        status, defects and notes.
        """
        logger.info("Exporting metadata to CSV")
        base_name = os.path.basename(self.input_dir.rstrip(os.sep))
        csv_path = os.path.join(self.output_dir, f"{base_name}.csv")
        meta = self._load_meta()
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            headers = [
                "Filename",
                "Tray/Directory",
                "Inspector",
                "Accept/Reject",
                "Defect(s)",
                "Notes",
            ]
            writer.writerow(headers)
            for fp in self.image_paths:
                bf = os.path.basename(fp)
                name, _ = os.path.splitext(bf)
                e = meta.get(bf, {})
                defects = e.get("defects", [])
                insp = e.get("inspector", "").strip()
                tray = e.get("tray", "").strip()
                note = e.get("note", "").strip()
                last_saved = e.get("last_saved", "").strip()
                if (not defects) and (last_saved in ("never", "")):
                    status = "Unlabeled"
                elif not defects:
                    status = "Accept"
                else:
                    status = "Reject"
                writer.writerow([name, tray, insp, status, ", ".join(defects), note])
        messagebox.showinfo("Export CSV", f"CSV saved to:\n{csv_path}")

    def _meta_path(self) -> str:
        """Return the path to the metadata JSON file."""
        return os.path.join(self.output_dir, "metadata.json")

    def _load_meta(self) -> Dict[str, Dict[str, str]]:
        """Load the metadata file if it exists, otherwise return an empty dict."""
        meta_path = self._meta_path()
        if os.path.isfile(meta_path):
            try:
                return json.load(open(meta_path))
            except Exception:
                return {}
        return {}

    # ---------------------------------------------------------------------
    # Logging helpers
    # ---------------------------------------------------------------------

    def _log_button(self, event: tk.Event) -> None:
        """Log button presses with widget context."""
        try:
            w = event.widget
            desc = None
            try:
                text = w.cget("text")
                if text:
                    desc = f"'{text}' button"
            except Exception:
                pass
            if not desc:
                desc = w.winfo_class()
            logger.info(f"{desc} clicked at ({event.x}, {event.y})")
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # Settings callbacks
    # ---------------------------------------------------------------------

    def choose_input_folder(self) -> None:
        """Open a dialog to select a new input folder and update configuration."""
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            config["Paths"]["input_folder"] = folder
            save_config()
            messagebox.showinfo("Config Saved", f"Input folder set to:\n{folder}")

    def choose_defects_config(self) -> None:
        """Open a dialog to select a new defects JSON file and update configuration."""
        fpath = filedialog.askopenfilename(
            title="Select Defects JSON", filetypes=[("JSON", "*.json")]
        )
        if fpath:
            config["Paths"]["defects_config"] = fpath
            save_config()
            messagebox.showinfo("Config Saved", f"Defects config set to:\n{fpath}")

    def choose_output_subdir(self) -> None:
        """Prompt for an output subdirectory name and update configuration."""
        sub = simpledialog.askstring(
            "Output Subdirectory",
            "Enter output subdirectory name:",
            initialvalue=config["Paths"].get("output_subdir", "output"),
        )
        if sub:
            config["Paths"]["output_subdir"] = sub
            save_config()
            # Update runtime directories
            self.input_dir = config["Paths"]["input_folder"]
            self.output_dir = os.path.join(
                self.input_dir, config["Paths"].get("output_subdir", "output")
            )
            os.makedirs(self.output_dir, exist_ok=True)
            messagebox.showinfo("Config Saved", f"Output subdirectory set to:\n{sub}")

    def choose_model(self) -> None:
        """Open a dialog to select a new model checkpoint and reload predictions."""
        fpath = filedialog.askopenfilename(
            title="Select Model (.pth) File",
            filetypes=[("PyTorch Model", "*.pth *.pt"), ("All files", "*.*")],
        )
        if fpath:
            # Update the model path and persist to the config file
            self.model_path = fpath
            config["Paths"]["model_path"] = fpath
            save_config()
            # Clear any previously cached predictions; they were
            # generated using a different model and are no longer valid.
            self.pred_cache.clear()
            # Preload predictions for the new model.  Show a progress
            # indicator to inform the user that work is being done.  This
            # makes subsequent navigation responsive because inference
            # results are cached ahead of time.
            try:
                # Create a temporary progress bar for preloading
                self._progress_var = tk.DoubleVar()
                self._progress_bar = ttk.Progressbar(
                    self.root,
                    orient="horizontal",
                    length=300,
                    mode="determinate",
                    variable=self._progress_var,
                )
                self._progress_bar.pack(pady=10)
                self.root.update()
                self._preload_predictions()
            finally:
                # Remove the progress bar once caching is complete
                try:
                    self._progress_bar.destroy()
                except Exception:
                    pass
            messagebox.showinfo("Model Loaded", f"Model set to:\n{fpath}")
            self._progress_var.set(0)
            self._progress_bar = ttk.Progressbar(
                self.root,
                orient="horizontal",
                length=300,
                mode="determinate",
                variable=self._progress_var,
            )
            self._progress_bar.pack(pady=10)
            self.root.update()
            self._preload_predictions()
            self._progress_bar.destroy()
            # Refresh inference on the current image
            if self.on_image_change:
                self.on_image_change(self.image_paths[self.idx], self.layer)

    # ---------------------------------------------------------------------
    # Search functionality
    # ---------------------------------------------------------------------

    def _on_search_submit(self, event: Optional[tk.Event] = None) -> None:
        """
        Jump to the first image whose filename contains the search string
        entered into the search box.
        """
        query = self.search_var.get().strip().lower()
        for i, path in enumerate(self.image_paths):
            filename = os.path.basename(path).lower()
            if query in filename:
                self.idx = i
                self._load_image()
                self._fit_image_to_canvas()
                self._show_image()
                if self.on_image_change:
                    self.on_image_change(self.image_paths[self.idx], self.layer)
                return
        messagebox.showinfo("Not Found", f"No image matching: {query}")

    # ---------------------------------------------------------------------
    # Inference status update
    # ---------------------------------------------------------------------

    def update_infer_info(
        self, status: str, confidence: str, cam_pil: Optional[Image.Image]
    ) -> None:
        """
        Update the inference result and confidence displayed in the UI.

        Parameters
        ----------
        status: str
            Predicted class label (``'Accept'`` or ``'Reject'``).
        confidence: str
            Probability of the predicted class as a percentage (e.g. '85.4%').
        cam_pil: Optional[Image.Image]
            Currently unused placeholder for future enhancements.
        """
        self.infer_result_var.set(f"Result: {status}")
        self.model_status = status
        self.infer_conf_var.set(f"Confidence: {confidence}")
        # Record numeric confidence for metadata export
        try:
            # Remove the trailing '%' sign if present
            if confidence.endswith("%"):
                self.model_confidence = float(confidence[:-1])
            else:
                self.model_confidence = float(confidence)
        except Exception:
            self.model_confidence = 0.0


# ---------------------------------------------------------------------------
# Inference viewer
# ---------------------------------------------------------------------------


class InferenceViewer:
    """
    A simple viewer for displaying original images, Grad‑CAM heatmaps,
    overlays and combined masks side by side.  Users can zoom and pan
    simultaneously across all views.  A text widget at the bottom
    displays a summary of the model inference.

    The viewer does not run inference itself; it simply displays
    arrays provided by the caller.

    Parameters
    ----------
    parent: tk.Widget
        The Tkinter parent in which to build the viewer.
    model_path: str
        Path to the model checkpoint (used only by the parent application
        to run inference; not used directly by this class).
    """

    def __init__(
        self, parent: tk.Widget, model_path: str = "best_val_loss_model.pth"
    ) -> None:
        self.parent = parent
        self.model_path = Path(model_path)
        # Shared pan offsets for zoom/pan across all canvases
        self.pan_x: int = 0
        self.pan_y: int = 0
        # Per‑view state: canvases, zoom factors, current image IDs, PIL images
        self.all_c: Dict[str, tk.Canvas] = {}
        self.all_zoom: Dict[str, float] = {}
        self.all_id: Dict[str, Optional[int]] = {}
        self.pil_imgs: Dict[str, Image.Image] = {}
        # Build the UI
        self._build_ui()

    def _build_ui(self) -> None:
        """Create the notebook and canvases for displaying the various views."""
        # Notebook with a single tab containing all views
        nb = ttk.Notebook(self.parent)
        nb.pack(expand=True, fill="both")
        container = ttk.Frame(nb)
        canvas = tk.Canvas(container, highlightthickness=0)
        vsb = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)
        vsb.pack(side="right", fill="y", expand=False)
        hsb.pack(side="bottom", fill="both", expand=False)
        canvas.pack(side="left", fill="both", expand=True)
        all_frame = ttk.Frame(canvas)
        # Anchor the inner frame to the top-left so that widgets are placed correctly
        canvas.create_window((0, 0), window=all_frame, anchor="nw")
        all_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        nb.add(container, text="All Views")
        # Grid configuration: two rows of images, one row for summary
        # Configure row and column weights so that views expand uniformly
        for r in range(5):
            all_frame.grid_rowconfigure(r, weight=1)
        for c in range(3):
            all_frame.grid_columnconfigure(c, weight=1)
        # View titles
        titles = ("Original", "Heatmap", "Overlay", "Mask Overlay")
        self.all_labels: Dict[str, ttk.Label] = {}
        for i, title in enumerate(titles):
            row, col = divmod(i, 2)
            lbl = ttk.Label(all_frame, text=title)
            lbl.grid(row=row * 2, column=col, sticky="n", padx=5, pady=(5, 0))
            self.all_labels[title] = lbl
            c = tk.Canvas(all_frame, bg="black", highlightthickness=0)
            c.grid(row=row * 2 + 1, column=col, sticky="nsew", padx=5, pady=(0, 5))
            self.all_c[title] = c
            self.all_zoom[title] = 1.0
            self.all_id[title] = None
            # Bind zoom and pan
            c.bind("<MouseWheel>", lambda e, t=title: self._on_zoom(e, t))
            c.bind("<Button-4>", lambda e, t=title: self._on_zoom(e, t))
            c.bind("<Button-5>", lambda e, t=title: self._on_zoom(e, t))
            c.bind("<ButtonPress-3>", lambda e, t=title: self._on_pan_start(e))
            c.bind("<B3-Motion>", lambda e, t=title: self._on_pan_move(e, t))
        # Summary text area
        self.txt = tk.Text(all_frame, wrap="word", state="disabled")
        self.txt.grid(row=0, column=2, rowspan=4, sticky="nsew", padx=10)

    def _on_zoom(self, event: tk.Event, view: str) -> None:
        """Zoom all views in or out synchronously."""
        factor = (
            1.1
            if (getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4)
            else 0.9
        )
        for v in self.all_zoom:
            self.all_zoom[v] = max(0.1, min(self.all_zoom[view] * factor, 10.0))
        self.pan_x = self.pan_y = 0
        for v in self.all_c:
            self._refresh_view(v)

    def _on_pan_start(self, event: tk.Event) -> None:
        self._prx, self._pry = event.x, event.y

    def _on_pan_move(self, event: tk.Event, view: str) -> None:
        dx, dy = event.x - self._prx, event.y - self._pry
        self._prx, self._pry = event.x, event.y
        self.pan_x += dx
        self.pan_y += dy
        for v in self.all_c:
            self._refresh_view(v)

    def _refresh_view(self, view: str) -> None:
        """Redraw a single view given the current zoom and pan settings."""
        pil = self.pil_imgs.get(view)
        if pil is None:
            return
        z = self.all_zoom[view]
        w, h = int(pil.width * z), int(pil.height * z)
        w, h = max(w, 1), max(h, 1)
        resized = pil.resize((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        c = self.all_c[view]
        c.photo = photo  # Prevent garbage collection
        c.delete("all")
        self.all_id[view] = c.create_image(
            self.pan_x,
            self.pan_y,
            anchor="nw",
            image=photo,
        )
        c.config(scrollregion=c.bbox("all"))

    def display(
        self,
        orig: np.ndarray,
        heat: np.ndarray,
        ov: np.ndarray,
        summary: str,
        mask_layer: Optional[Image.Image] = None,
    ) -> None:
        """
        Update the viewer with new images and a summary.  The heatmap is
        suppressed when the prediction is ``Accept``.

        Parameters
        ----------
        orig: np.ndarray
            Original image array.
        heat: np.ndarray
            Heatmap array.
        ov: np.ndarray
            Overlay array from Grad‑CAM blending.
        summary: str
            Human‑readable summary string from :func:`run_inference`.
        mask_layer: Optional[Image.Image]
            Current annotation overlay; if provided it is composited with
            the overlay image in a fourth view.
        """
        pred_line = next(
            (l for l in summary.splitlines() if l.startswith("Predicted class")), ""
        )
        pred = pred_line.split(":", 1)[1].strip() if ":" in pred_line else ""
        # Hide the heatmap view when Accept
        for title in self.all_labels:
            self.all_labels[title].grid()
            self.all_c[title].grid()
        imgs: Dict[str, Image.Image] = {
            "Original": Image.fromarray(orig),
            "Overlay": Image.fromarray(ov),
        }
        if pred == "Reject":
            # convert heatmap to PIL then resize to other views
            heat_pil = Image.fromarray(heat)
            heat_pil = heat_pil.resize(imgs["Original"].size, Image.LANCZOS)
            imgs["Heatmap"] = heat_pil
        else:
            # Hide heatmap views on Accept
            self.all_labels["Heatmap"].grid_remove()
            self.all_c["Heatmap"].grid_remove()
        # Add mask overlay if provided
        if mask_layer:
            base_ov = imgs["Overlay"].convert("RGBA")
            comp = Image.alpha_composite(base_ov, mask_layer)
            imgs["Mask Overlay"] = comp.convert("RGB")

        # Compute a single “initial_zoom” based on the Original view
        orig_pil = imgs["Original"]
        orig_canvas = self.all_c["Original"]
        cw, ch = max(orig_canvas.winfo_width(), 1), max(orig_canvas.winfo_height(), 1)
        base_zoom_orig = min(cw / orig_pil.width, ch / orig_pil.height, 1.0)
        initial_zoom = base_zoom_orig
        self.pan_x = self.pan_y = 0
        # Apply same zoom to every view
        for title, pil in imgs.items():
            self.all_zoom[title] = initial_zoom
            self.pil_imgs[title] = pil
            self._refresh_view(title)

        # Update the summary text
        self.txt.config(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", summary)
        self.txt.config(state="disabled")


# ---------------------------------------------------------------------------
# Application integrator
# ---------------------------------------------------------------------------


class AnnoMateApp:
    """
    Combine the MaskingTool and InferenceViewer into a single application
    with tabbed panes.  On initialisation the user is prompted for an
    input directory (if not already set in the configuration), and the
    two tools are synchronised such that changes in one are reflected in
    the other.
    """

    def __init__(self) -> None:
        # Create the top‑level window
        self.root = tk.Tk()
        self.root.title("AnnoMate – Masking and Inference Tool")
        self.root.geometry("1200x800")
        # Read configuration to determine paths
        config_file = CONFIG_PATH
        cfg = configparser.ConfigParser()
        cfg.read(config_file)
        if "Paths" not in cfg:
            cfg["Paths"] = {}
        # Determine input folder
        input_folder = cfg["Paths"].get("input_folder", "")
        if not input_folder or not os.path.isdir(input_folder):
            selected = filedialog.askdirectory(title="Select input folder")
            if not selected:
                messagebox.showerror(
                    "No input folder", "An input folder is required. Exiting."
                )
                sys.exit(1)
            cfg["Paths"]["input_folder"] = selected
            with open(config_file, "w") as f:
                cfg.write(f)
            input_folder = selected
        defects_cfg = cfg["Paths"].get("defects_config", "defects_config.json")
        dataset = cfg["Paths"].get("dataset", "bowtie")
        output_folder = os.path.join(
            input_folder, cfg["Paths"].get("output_subdir", "output")
        )
        os.makedirs(output_folder, exist_ok=True)
        # Create a notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)
        # Masking tab
        mask_frame = ttk.Frame(notebook)
        notebook.add(mask_frame, text="Masking Tool")
        self.mask = MaskingTool(
            mask_frame, input_folder, output_folder, defects_cfg, dataset
        )
        # Inference tab
        infer_frame = ttk.Frame(notebook)
        notebook.add(infer_frame, text="Inference Viewer")
        self.infer = InferenceViewer(infer_frame, model_path="best_val_loss_model.pth")

        # Synchronise the two tools
        def on_change(path: str, mask_layer: Image.Image) -> None:
            """
            Synchronise the inference viewer and masking tool whenever the
            current image or overlay changes.  This handler first checks
            whether predictions have been precomputed for the given path
            using the current model.  If cached results exist, they are
            used directly to avoid rerunning inference and reduce lag.  If
            no cache entry is available, inference is performed and, on
            success, the results are inserted into the cache.

            Parameters
            ----------
            path: str
                Full file path of the image to infer upon.
            mask_layer: Image.Image
                The current annotation overlay to composite with the
                inference overlay.
            """
            # Ensure that the inference viewer knows which model is being
            # used.  This is primarily for informational purposes within
            # the viewer; the actual inference is handled here.
            self.infer.model_path = Path(self.mask.model_path)
            cached = None
            # Check the prediction cache for this path and model.  The
            # cache is keyed only on the image path; whenever the model
            # changes the cache is cleared in `choose_model`, so it is
            # safe to use this simple lookup.
            try:
                cached = self.mask.pred_cache.get(path)
            except Exception:
                cached = None
            if cached:
                orig, heat, ov, summary, pred, prob = cached
            else:
                model_fp = str(self.mask.model_path)
                try:
                    orig, heat, ov, summary, pred, prob = run_inference(model_fp, path)
                    # Store successful results in the cache for future use
                    self.mask.pred_cache[path] = (orig, heat, ov, summary, pred, prob)
                except Exception as ex:
                    # If inference fails, log the error and produce a
                    # minimal fallback so the UI remains responsive.
                    logger.error(f"Inference failed for {path}: {ex}")
                    try:
                        img_bgr = cv2.imread(path)
                        orig = (
                            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            if img_bgr is not None
                            else np.zeros((10, 10, 3), np.uint8)
                        )
                        if img_bgr is None:
                            raise FileNotFoundError(path)
                    except Exception:
                        # Last resort, create small dummy image
                        orig = np.zeros((10, 10, 3), np.uint8)
                    ov = orig.copy()
                    heat = np.zeros_like(orig)
                    summary = f"Inference failed: {ex}"
                    pred = "N/A"
                    prob = None
            # Determine the predicted class and compute a formatted
            # confidence percentage.  When cached results exist, use the
            # cached prediction and probability instead of parsing the
            # summary string, as the latter may be expensive for frequent
            # navigation events.
            status = "N/A"
            confidence = "N/A"
            try:
                if cached:
                    status = pred
                    if prob is not None:
                        confidence = f"{prob * 100:.1f}%"
                else:
                    # Parse the summary string for human‑readable status and
                    # confidence information.  This is preserved for
                    # compatibility with existing UI expectations.
                    lines = summary.splitlines()
                    pred_line = next(
                        (l for l in lines if l.startswith("Predicted class")), ""
                    )
                    prob_line = next(
                        (l for l in lines if l.startswith("Probabilities")), ""
                    )
                    status = (
                        pred_line.split(":", 1)[1].strip()
                        if ":" in pred_line
                        else "N/A"
                    )
                    if ":" in prob_line:
                        if status == "Accept":
                            raw = prob_line.split("Accept=", 1)[1].split("|")[0]
                        else:
                            raw = prob_line.split("Reject=", 1)[1]
                        val = float(raw)
                        confidence = f"{val * 100:.1f}%"
            except Exception:
                # Fall back to defaults if parsing fails
                status = pred if pred else "N/A"
                if prob is not None:
                    confidence = f"{prob * 100:.1f}%"
            # Update the MaskingTool’s inference panel with the status and
            # confidence values.  Convert the overlay to a PIL image for
            # consistency with the rest of the UI code.
            self.mask.update_infer_info(status, confidence, Image.fromarray(ov))
            # Display the images and summary in the inference viewer.  The
            # mask_layer argument is forwarded so the viewer can
            # composite the current annotations with the inference
            # overlay.  Note: when cached results are used the summary
            # string remains unchanged from the cached value.
            self.infer.display(orig, heat, ov, summary, mask_layer)

        # Set callback and invoke for the initial image
        initial = self.mask.image_paths[self.mask.idx]
        self.mask.on_image_change = on_change
        on_change(initial, self.mask.layer)
        # Close gracefully
        try:
            self.root.winfo_toplevel().protocol("WM_DELETE_WINDOW", self.on_close)
        except Exception:
            pass

    def on_close(self) -> None:
        """Shutdown handler for the application."""
        try:
            self.mask._on_close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        """Start the Tkinter main loop."""
        self.root.mainloop()


if __name__ == "__main__":
    # Launch the application when run directly
    app = AnnoMateApp()
    app.run()
