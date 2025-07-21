import sys
import os
import json


def resource_path(relative_path):
    """Return the path to a bundled resource, whether running + inside PyInstaller or in a normal Python environment"""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)


import argparse
import numpy as np
import cv2
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
from gui_infer import run_inference
from PIL.ImageDraw import floodfill
from openpyxl import Workbook
import csv
import glob
import configparser
from pathlib import Path

from style_config import (
    COLUMN_WEIGHTS,
    ROW_WEIGHTS,
    PAD_SMALL,
    PAD_MEDIUM,
    PAD_LARGE,
    BUTTON_FONT,
    LABEL_FONT,
    ENTRY_FONT,
    BUTTON_PAD,
    CANVAS_BG,
    CANVAS_CURSOR,
    TS_DISPLAY_FMT,
    TS_FILENAME_FMT,
    PEN1_COLOR,
    PEN2_COLOR,
    PEN3_COLOR,
    PEN4_COLOR,
    PEN5_COLOR,
    PEN6_COLOR,
)

import logging
from logging.handlers import RotatingFileHandler

# ─── Application Logging Setup ──────────────────────────────────────────────────

logger = logging.getLogger("AnnoMate")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = RotatingFileHandler(
    "AnnoMate.log", maxBytes=5 * 1024 * 1024, backupCount=2
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ─── Configuration for user paths ─────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.ini")
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


def save_config():
    """Save configuration to config.ini"""
    with open(CONFIG_PATH, "w") as cfg_file:
        config.write(cfg_file)


class MaskingTool:

    def _refresh_pen_widgets(self):
        # Hide all widgets
        for widgets in self.pen_widgets.values():
            for w in widgets:
                w.grid_remove()

        # Show widgets for selected checkboxes
        visible_keys = set()
        mapping = {
            "chip": "pen1",
            "scratch": "pen2",
            "gouge": "pen3",
            "inclusion": "pen4",
            "void": "pen5",
            "other": "pen6",
        }
        for defect, var in self.defect_vars.items():
            if var.get() and defect in mapping:
                visible_keys.add(mapping[defect])

        for i, key in enumerate(sorted(visible_keys)):
            label, combo, btn = self.pen_widgets[key]
            label.grid(row=2 * i, column=0, sticky="w", padx=PAD_SMALL)
            combo.grid(row=2 * i, column=1, sticky="ew", padx=PAD_SMALL)
            btn.grid(
                row=2 * i + 1,
                column=0,
                columnspan=2,
                sticky="w",
                padx=PAD_SMALL,
                pady=(0, 5),
            )

        self.pen_frame.columnconfigure(1, weight=1)

    def _log_interaction(self, e):
        """
        Log button clicks with human-friendly names.
        Detects ttk.Button or tk.Button and logs its text label.
        """
        w = e.widget
        desc = None
        # Try to get widget text for buttons
        try:
            text = w.cget("text")
            if text:
                desc = f"'{text}' button"
        except Exception:
            pass
        if not desc:
            # Fallback to widget class name
            desc = w.winfo_class()
        logger.info(f"{desc} clicked at ({e.x}, {e.y})")

    def __init__(self, parent, input_dir, output_dir, config_path, dataset):
        self.model_status = ""
        self.model_confidence = 0.0
        self.model_path = "best_val_acc_model.pth"
        self.on_image_change = None
        self.root = parent
        self.input_dir = input_dir
        self.output_dir = output_dir
        self._zoom_after_id = None

        # ensure the output folder exists
        os.makedirs(self.output_dir, exist_ok=True)

        # spin up a second handler writing into output/AnnoMate.log
        log_path = os.path.join(self.output_dir, "AnnoMate.log")
        fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=2)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Load defect list from JSON config
        cfg_path = resource_path(config_path)
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        self.defects = cfg.get(dataset, cfg.get("bowtie", []))

        # Gather image paths
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        self.image_paths = sorted(
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.splitext(f.lower())[1] in exts
        )
        if not self.image_paths:
            raise RuntimeError(f"No images in {input_dir}")

        # Annotation state
        self.idx = 0
        self.mode = "draw"
        self.history = []
        self.redo_stack = []
        self.pan_x = self.pan_y = 0
        self.zoom = 1.0
        self.drawing = False

        # Pen state
        self.current_pen = "pen1"
        self.pen_color = PEN1_COLOR
        self.pen_width = 8
        self._tk_pen_color = "#%02x%02x%02x" % self.pen_color[:3]
        self._current_stroke_ids = []

        # Build UI
        self._build_ui()

        # Bind defect trace callbacks to remove marks on uncheck
        for defect, var in self.defect_vars.items():
            var.trace_add(
                "write", lambda *args, defect=defect: self._on_defect_toggle(defect)
            )

        # capture clicks, keypresses, window-resize events, etc.
        self.root.bind_all("<ButtonPress>", self._log_interaction)
        self.root.bind_all(
            "<KeyPress>",
            lambda e: logger.info(f"Key {e.keysym} pressed in {e.widget}"),
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
            self.root.winfo_toplevel().protocol("WM_DELETE_WINDOW", self.on_close)
        except Exception:
            pass

        # Load first image
        self._load_image()
        self.root.update()
        self._fit_image_to_canvas()
        self._show_image()
        if self.on_image_change:
            self.on_image_change(self.image_paths[self.idx], self.layer)

    # ─── New methods for Settings menu ──────────────────────────────────────────
    def choose_input_folder(self):
        """Open dialog to select input folder and save to config"""
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            config["Paths"]["input_folder"] = folder
            save_config()
            messagebox.showinfo("Config Saved", f"Input folder set to:\n{folder}")

    def choose_defects_config(self):
        """Open dialog to select defects JSON and save to config"""
        fpath = filedialog.askopenfilename(
            title="Select Defects JSON", filetypes=[("JSON", "*.json")]
        )
        if fpath:
            config["Paths"]["defects_config"] = fpath
            save_config()
            messagebox.showinfo("Config Saved", f"Defects config set to:\n{fpath}")

    def choose_output_subdir(self):
        """Prompt for output subdirectory name and save to config"""
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

    def _wrap_cmd(self, fn):
        # Wrapper to disable fill mode before executing the given function
        def wrapped(*args, **kwargs):
            self.fill_var.set(False)
            return fn(*args, **kwargs)

        return wrapped

    def _build_ui(self):
        try:
            self.root.winfo_toplevel().title("AnnoMate")
        except Exception:
            pass

        # ─── set application icon ────────────────────────────────────────────────
        # build the path to your AnnoMate.ico inside the tmp MEIPASS folder or
        # alongside your script (whichever is in use)
        base_path = getattr(sys, "_MEIPASS", os.path.dirname(__file__))
        ico_path = os.path.join(base_path, "imgs", "AnnoMate.ico")

        try:
            # On Windows this should load a true .ico file as the window icon.
            # If it fails (as it did in your frozen exe), it will jump to the except block.
            self.root.winfo_toplevel().iconbitmap(ico_path)
        except tk.TclError:
            # Fallback: load the .ico via PIL.Image and set via iconphoto,
            # which always works on both Windows and macOS.
            #
            # 1) open the .ico with PIL
            img = Image.open(ico_path)
            # 2) convert it into a Tk PhotoImage
            self._tk_icon = ImageTk.PhotoImage(img)
            # 3) attach it to the window (True=apply to window and any toplevels)
            self.root.iconphoto(True, self._tk_icon)
        # ───────────────────────────────────────────────────────────────────────────

        # ─── Model lookup & initial run ───────────────────────────────────────────
        base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        self.model_path = Path(base) / self.model_path
        if not self.model_path.exists():
            messagebox.showerror("Error", f"Model not found: {self.model_path}")
            return

        # ─── Settings menu for configuration ─────────────────────────────────
        self.menu = tk.Menu(self.root)
        menubar = self.menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(
            label="Input Folder…", command=self.choose_input_folder
        )
        settings_menu.add_command(
            label="Defects JSON…", command=self.choose_defects_config
        )
        settings_menu.add_command(
            label="Output Subdir…", command=self.choose_output_subdir
        )
        menubar.add_cascade(label="Settings", menu=settings_menu)


        # Styles
        style = ttk.Style(self.root)
        style.configure("App.TButton", font=BUTTON_FONT)
        style.configure("App.TLabel", font=LABEL_FONT)
        style.configure("App.TEntry", font=ENTRY_FONT)
        style.configure(
            "Fill.Toolbutton",
            padding=(5, 5),
            background="#d9d9d9",
            foreground="#000000",
            borderwidth=2,
            relief="raised",
            font=("Arial", 14),
        )
        style.map(
            "Fill.Toolbutton",
            background=[("active", "#bfbfbf"), ("selected", "#2c69c4")],
            relief=[("selected", "sunken"), ("!selected", "raised")],
            foreground=[("selected", "#000000"), ("!selected", "#000000")],
        )

        # Layout weights
        for col, w in COLUMN_WEIGHTS.items():
            self.root.columnconfigure(col, weight=w)
        for row, w in ROW_WEIGHTS.items():
            self.root.rowconfigure(row, weight=w)

        # Canvas for image and drawing
        # Paned window to allow resizing of the control panel
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.canvas = tk.Canvas(self.root, bg=CANVAS_BG, cursor=CANVAS_CURSOR)
        self.paned.add(self.canvas)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<ButtonPress-3>", self._on_right_press)
        self.canvas.bind("<B3-Motion>", self._on_right_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_release)
        self.canvas.bind(
            "<MouseWheel>", lambda ev: self._wrap_cmd(self._on_mousewheel)(ev)
        )
        self.canvas.bind(
            "<Button-4>", lambda ev: self._wrap_cmd(lambda e: self._set_zoom(1.1))(ev)
        )
        self.canvas.bind(
            "<Button-5>", lambda ev: self._wrap_cmd(lambda e: self._set_zoom(0.9))(ev)
        )

        # Control panel with scrollbar
        # Control panel container now lives in the PanedWindow
        ctrl_container = ttk.Frame(self.paned, padding=(0, 0))
        # Logo
        logo_path = os.path.join(os.path.dirname(__file__), "imgs", "AnnoMate.png")
        try:
            logo_img = Image.open(logo_path)
            logo_img = logo_img.resize((100, 100), Image.LANCZOS)
            self.logo_tk = ImageTk.PhotoImage(logo_img)
            logo_label = ttk.Label(ctrl_container, image=self.logo_tk)
            logo_label.pack(side="top", anchor="center", pady=(0, PAD_SMALL))
        except Exception as e:
            print(f"Failed to load logo: {e}")

        self.paned.add(ctrl_container)
        self.ctrl_canvas = tk.Canvas(
            ctrl_container, borderwidth=0, highlightthickness=0
        )
        vsb = ttk.Scrollbar(
            ctrl_container, orient="vertical", command=self.ctrl_canvas.yview
        )
        hsb = ttk.Scrollbar(
            ctrl_container, orient=tk.HORIZONTAL, command=self.ctrl_canvas.xview
        )
        self.ctrl_canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.ctrl_frame = ttk.Frame(self.ctrl_canvas)
        self.ctrl_canvas.create_window((0, 0), window=self.ctrl_frame, anchor="nw")
        self.ctrl_canvas.pack(side="top", fill="both", expand=True, pady=(0, 0))
        self.ctrl_frame.bind(
            "<Configure>",
            lambda e: self.ctrl_canvas.configure(
                scrollregion=self.ctrl_canvas.bbox("all")
            ),
        )

        # somewhere in your _build_ui() after creating self.ctrl_frame:
        settings_btn = tk.Menubutton(self.ctrl_frame, text="⚙", relief="raised")
        settings_btn.grid(row=0, column=0, sticky="ne", padx=5, pady=5)
        settings_menu = tk.Menu(settings_btn, tearoff=0)
        settings_btn["menu"] = settings_menu

        settings_menu.add_command(label="Input Folder…",   command=self.choose_input_folder)
        settings_menu.add_command(label="Defects JSON…",   command=self.choose_defects_config)
        settings_menu.add_command(label="Output Subdir…",  command=self.choose_output_subdir)

        # ─── enable scroll-wheel on control panel ─────────────────────────────────

        # when mouse enters the control frame, bind all wheel events to our handler
        self.ctrl_frame.bind(
            "<Enter>",
            lambda e: self.ctrl_canvas.bind_all(
                "<MouseWheel>", self._on_control_panel_wheel
            ),
        )
        # when it leaves, unbind so wheel goes back to default
        self.ctrl_frame.bind(
            "<Leave>", lambda e: self.ctrl_canvas.unbind_all("<MouseWheel>")
        )

        # (Optional for Linux scrollpads)
        self.ctrl_frame.bind(
            "<Enter>",
            lambda e: (
                self.ctrl_canvas.bind_all("<Button-4>", self._on_control_panel_wheel),
                self.ctrl_canvas.bind_all("<Button-5>", self._on_control_panel_wheel),
            ),
        )
        self.ctrl_frame.bind(
            "<Leave>",
            lambda e: (
                self.ctrl_canvas.unbind_all("<Button-4>"),
                self.ctrl_canvas.unbind_all("<Button-5>"),
            ),
        )

        # ─── global wheel binding ──────────────────────────────────────────────────
        # anywhere in the window, catch all wheel events...
        self.root.bind_all("<MouseWheel>", self._on_global_wheel)
        self.root.bind_all("<Button-4>", self._on_global_wheel)  # Linux scroll up
        self.root.bind_all("<Button-5>", self._on_global_wheel)  # Linux scroll down

        # Variables for metadata
        self.note_var = tk.StringVar(master=self.root)
        # create a single StringVar for inspector
        self.inspector_var = tk.StringVar(master=self.root)
        # whenever it changes, call the handler
        # (trace_add gives you name, index, mode args; your handler can accept *args)

        self.tray_var = tk.StringVar(master=self.root)
        self.filename_var = tk.StringVar(master=self.root)
        self.save_ts_var = tk.StringVar(master=self.root, value="never")
        self.fill_var = tk.BooleanVar(master=self.root)

        # Navigation buttons
        nav_frame = ttk.LabelFrame(
            self.ctrl_frame, text="Main Controls", padding=(5, 5)
        )
        nav_frame.grid(row=0, column=0, sticky="ew")
        actions = [
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
            btn = ttk.Button(
                nav_frame, text=txt, command=self._wrap_cmd(fn), style="App.TButton"
            )
            btn.grid(
                row=i // 2,
                column=i % 2,
                sticky="ew",
                padx=BUTTON_PAD[0],
                pady=BUTTON_PAD[1],
            )
        for c in range(2):
            nav_frame.columnconfigure(c, weight=1)

        # Zoom controls
        zoom_frame = ttk.LabelFrame(self.ctrl_frame, text="Zoom", padding=(5, 5))
        zoom_frame.grid(row=1, column=0, sticky="ew")
        for symbol, factor in [("−", 0.9), ("+", 1.1)]:
            ttk.Button(
                zoom_frame,
                text=symbol,
                command=self._wrap_cmd(lambda f=factor: self._set_zoom(f)),
                style="App.TButton",
            ).pack(side="left", padx=BUTTON_PAD[0], pady=BUTTON_PAD[1])

        # Metadata display
        meta_frame = ttk.LabelFrame(self.ctrl_frame, text="Information", padding=(5, 5))
        meta_frame.grid(row=2, column=0, sticky="ew")
        # Note field
        ttk.Label(meta_frame, text="Note:").grid(row=0, column=0, sticky="nw", padx=5)
        self.note_text = tk.Text(meta_frame, wrap="word", font=ENTRY_FONT, height=3)
        self.note_text.grid(row=0, column=1, sticky="ew", padx=5)
        self.note_text.bind("<KeyRelease>", lambda e: self._adjust_note_height())
        # Inspector combobox

        ttk.Label(meta_frame, text="Inspector:").grid(
            row=1, column=0, sticky="w", padx=5
        )
        self.inspector_entry = ttk.Entry(
            meta_frame, textvariable=self.inspector_var, style="App.TEntry"
        )
        self.inspector_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)

        # Tray
        ttk.Label(meta_frame, text="Tray:").grid(row=2, column=0, sticky="w", padx=5)
        ttk.Entry(
            meta_frame, textvariable=self.tray_var, style="App.TEntry", state="readonly"
        ).grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        # Filename
        ttk.Label(meta_frame, text="Filename:").grid(
            row=3, column=0, sticky="w", padx=5
        )
        ttk.Entry(
            meta_frame,
            textvariable=self.filename_var,
            style="App.TEntry",
            state="readonly",
        ).grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        # Last saved timestamp
        ttk.Label(meta_frame, text="Last saved:").grid(
            row=4, column=0, sticky="w", padx=5
        )
        ttk.Label(meta_frame, textvariable=self.save_ts_var).grid(
            row=4, column=1, sticky="w", padx=5, pady=2
        )
        meta_frame.columnconfigure(1, weight=1)

        # Categories (defects)
        def_frame = ttk.LabelFrame(self.ctrl_frame, text="Categories", padding=(5, 5))
        def_frame.grid(row=3, column=0, sticky="ew")
        self.defect_vars = {}
        for i, d in enumerate(self.defects):
            var = tk.BooleanVar(master=self.root)
            cb = ttk.Checkbutton(def_frame, text=d, variable=var)
            cb.grid(row=i, column=0, sticky="w", padx=PAD_SMALL)
            var.trace_add("write", self._update_pen_labels)
            self.defect_vars[d] = var

        # Fill mode toggle
        fill_frame = ttk.LabelFrame(self.ctrl_frame, text="Fill", padding=(5, 5))
        fill_frame.grid(row=4, column=0, sticky="ew")
        ttk.Checkbutton(
            fill_frame,
            text="PEN FILL MODE",
            variable=self.fill_var,
            style="Fill.Toolbutton",
        ).pack(padx=PAD_SMALL, pady=PAD_SMALL)

        # Pen Labels (6 pens)
        self.pen_frame = ttk.LabelFrame(
            self.ctrl_frame, text="Pen Labels", padding=(5, 5)
        )
        self.pen_frame.grid(row=5, column=0, sticky="ew")

        self.pen_vars = {
            "pen1": tk.StringVar(master=self.root),
            "pen2": tk.StringVar(master=self.root),
            "pen3": tk.StringVar(master=self.root),
            "pen4": tk.StringVar(master=self.root),
            "pen5": tk.StringVar(master=self.root),
            "pen6": tk.StringVar(master=self.root),
        }
        self.pen_widgets = {}
        pen_order = [
            ("pen1", "Red"),
            ("pen2", "Blue"),
            ("pen3", "Green"),
            ("pen4", "Yellow"),
            ("pen5", "Magenta"),
            ("pen6", "Cyan"),
        ]

        for i, (key, color) in enumerate(pen_order):
            label = ttk.Label(self.pen_frame, text=f"{key.upper()} label:")
            combo = ttk.Combobox(
                self.pen_frame,
                textvariable=self.pen_vars[key],
                state="readonly",
                style="App.TEntry",
                width=30,
            )
            btn = ttk.Button(
                self.pen_frame,
                text=f"Use {key.upper()} ({color})",
                command=self._wrap_cmd(lambda k=key: self.select_pen(k)),
                style="App.TButton",
            )
            self.pen_widgets[key] = (label, combo, btn)

        self._refresh_pen_widgets()

        # ─── Inference Info (Accept/Reject + Confidence + static Grad‑CAM) ────
        inf_frame = ttk.LabelFrame(self.ctrl_frame, text="Inference", padding=(5, 5))
        inf_frame.grid(row=7, column=0, sticky="ew", pady=(0, PAD_SMALL))
        # Status & confidence
        self.infer_result_var = tk.StringVar(master=self.root, value="Result: N/A")
        self.infer_conf_var = tk.StringVar(master=self.root, value="Confidence: N/A")
        ttk.Label(inf_frame, textvariable=self.infer_result_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(inf_frame, textvariable=self.infer_conf_var).grid(
            row=1, column=0, sticky="w"
        )

        # Export buttons
        export_frame = ttk.LabelFrame(
            self.ctrl_frame, text="Export Information", padding=(5, 5)
        )
        export_frame.grid(row=6, column=0, sticky="ew")
        ttk.Button(
            export_frame,
            text="Export as XLSX",
            command=self._wrap_cmd(self.export_excel),
            style="App.TButton",
        ).pack(side="left", padx=PAD_SMALL)
        ttk.Button(
            export_frame,
            text="Export as CSV",
            command=self._wrap_cmd(self.export_csv),
            style="App.TButton",
        ).pack(side="left", padx=PAD_SMALL)
        

    def update_infer_info(self, status, confidence, cam_pil):
        self.infer_result_var.set(f"Result: {status}")
        self.model_status = status
        self.infer_conf_var.set(f"Confidence: {confidence}")

    def _on_defect_toggle(self, defect):
        # If defect was unchecked, remove its marks from the drawing layer
        if not self.defect_vars[defect].get():
            self._remove_defect_marks(defect)
            self._show_image()
            if self.on_image_change:
                self.on_image_change(self.image_paths[self.idx], self.layer)

    def _remove_defect_marks(self, defect):
        # Erase all pixels matching this defect's pen color
        # Map defect name to pen color
        pen_to_color = {
            "chip": PEN1_COLOR,
            "scratch": PEN2_COLOR,
            "gouge": PEN3_COLOR,
            "inclusion": PEN4_COLOR,
            "void": PEN5_COLOR,
            "other": PEN6_COLOR,
        }
        color = pen_to_color.get(defect)
        if color is None:
            return
        # Tolerance for color matching
        COLOR_TOL = 10
        w, h = self.base.size
        layer_pixels = self.layer.load()
        for x in range(w):
            for y in range(h):
                r, g, b, a = layer_pixels[x, y]
                if a == 0:
                    continue
                pr, pg, pb = color[:3]
                if (
                    abs(r - pr) <= COLOR_TOL
                    and abs(g - pg) <= COLOR_TOL
                    and abs(b - pb) <= COLOR_TOL
                ):
                    layer_pixels[x, y] = (0, 0, 0, 0)

    def _on_global_wheel(self, event):
        """
        If the cursor is over the image canvas → zoom;
        otherwise (i.e. it’s anywhere in the control panel side) → scroll the panel.
        """
        # get pointer in screen coords
        x, y = event.x_root, event.y_root

        # image‐canvas screen bbox
        ix1 = self.canvas.winfo_rootx()
        iy1 = self.canvas.winfo_rooty()
        ix2 = ix1 + self.canvas.winfo_width()
        iy2 = iy1 + self.canvas.winfo_height()

        if ix1 <= x <= ix2 and iy1 <= y <= iy2:
            # inside image → zoom
            if hasattr(event, "delta"):
                factor = 1.1 if event.delta > 0 else 0.9
            else:
                # Linux: Button-4 scroll up, 5 scroll down
                factor = 1.1 if event.num == 4 else 0.9
            self._set_zoom(factor)
        else:
            if hasattr(event, "delta"):
                # Windows & macOS: event.delta is ±120 per notch
                units = -1 if event.delta < 0 else 1
            else:
                # Linux: event.num 4=up, 5=down
                units = -1 if event.num == 4 else 1
        self.ctrl_canvas.yview_scroll(units, "units")

    def _on_control_panel_wheel(self, event):
        """
        Scroll the control‐panel canvas up/down when the mousewheel
        is used while the cursor is over the control frame.
        """
        # Windows & macOS: event.delta is ±120 per notch
        if hasattr(event, "delta") and event.delta:
            # negative delta → scroll down, positive → scroll up
            units = int(-1 * (event.delta / 120))
        else:
            # Linux: event.num 4=up, 5=down
            units = -1 if event.num == 4 else 1

        # Scroll the canvas
        self.ctrl_canvas.yview_scroll(units, "units")

    def _update_pen_labels(self, *args):
        # Fixed defect→pen assignment regardless of order
        mapping = {
            "chip": "pen1",
            "scratch": "pen2",
            "gouge": "pen3",
            "inclusion": "pen4",
            "void": "pen5",
            "other": "pen6",
        }
        # Clear all pens
        for k in self.pen_vars.keys():
            self.pen_vars[k].set("")
        # Assign each selected defect to its dedicated pen
        for defect, var in self.defect_vars.items():
            if var.get() and defect in mapping:
                self.pen_vars[mapping[defect]].set(defect)
        self._refresh_pen_widgets()
        # Fixed defect→pen assignment regardless of order
        mapping = {
            "chip": "pen1",
            "scratch": "pen2",
            "gouge": "pen3",
            "inclusion": "pen4",
            "void": "pen5",
            "other": "pen6",
        }
        # Clear all pens
        for k in self.pen_vars.keys():
            self.pen_vars[k].set("")
        # Assign each selected defect to its dedicated pen
        for defect, var in self.defect_vars.items():
            if var.get() and defect in mapping:
                self.pen_vars[mapping[defect]].set(defect)

    def _adjust_note_height(self, event=None):
        text = self.note_text
        line_count = int(text.index("end-1c").split(".")[0])
        text.configure(height=max(3, line_count))

    def select_pen(self, which):
        self.current_pen = which
        self.pen_color = {
            "pen1": PEN1_COLOR,
            "pen2": PEN2_COLOR,
            "pen3": PEN3_COLOR,
            "pen4": PEN4_COLOR,
            "pen5": PEN5_COLOR,
            "pen6": PEN6_COLOR,
        }[which]
        self.pen_width = 8
        self._tk_pen_color = "#%02x%02x%02x" % self.pen_color[:3]
        self.mode = "draw"
        self.canvas.config(cursor="pencil")

    def _load_image(self):
        full_path = self.image_paths[self.idx]
        logger.info(f"Loading image: {full_path}")
        self.base = Image.open(full_path).convert("RGB")

        folder = os.path.basename(os.path.dirname(full_path))
        file_name = os.path.basename(full_path)
        base_name, _ = os.path.splitext(file_name)

        # Tray and filename display
        self.tray_var.set(folder.split("_")[0])
        self.filename_var.set(f"{folder}/{file_name}")

        # ─── Load existing combined mask (if any) ────────────────────────────────
        # Reset transient UI state except inspector/tray/filename
        self.note_text.delete("1.0", "end")
        for v in self.defect_vars.values():
            v.set(False)
        for v in self.pen_vars.values():
            v.set("")
        masks_folder = os.path.join(self.output_dir, "masks")
        os.makedirs(masks_folder, exist_ok=True)

        # look for any "*-COMBINED-{base_name}.png" in masks_folder
        pattern = os.path.join(masks_folder, f"*-COMBINED-{base_name}.png")
        mask_files = glob.glob(pattern)

        if mask_files:
            # pick the most recently modified mask
            latest_mask = max(mask_files, key=os.path.getmtime)
            lm = Image.open(latest_mask).convert("RGBA")
            # if the saved mask size doesn’t match the base, resize it
            if lm.size != self.base.size:
                lm = lm.resize(self.base.size, Image.LANCZOS)
            self.layer = lm
        else:
            self.layer = Image.new("RGBA", self.base.size, (0, 0, 0, 0))
            # --- Write initial placeholder to disk immediately ---
            self._save_current()

        self.drawer = ImageDraw.Draw(self.layer)

        # Reset history/state
        self.history = [self.layer.copy()]
        self.redo_stack.clear()
        self.pan_x = self.pan_y = 0
        self.zoom = 1.0
        self.drawing = False

        # Load metadata
        key = os.path.basename(full_path)
        meta = self._load_meta().get(key, {})

        if "inspector" in meta:
            self.inspector_var.set(meta["inspector"])
        self.note_text.delete("1.0", "end")
        self.note_text.insert("1.0", meta.get("note", ""))

        if meta.get("tray"):
            self.tray_var.set(meta["tray"])

        for d, var in self.defect_vars.items():
            var.set(d in meta.get("defects", []))
        self._update_pen_labels()
        for pen_key, lab in meta.get("pen_labels", {}).items():
            if pen_key in self.pen_vars:
                self.pen_vars[pen_key].set(lab)
        self.save_ts_var.set(meta.get("last_saved", "never"))
        self._unsaved_clear = False

    def _show_image(self):
        comp = Image.alpha_composite(self.base.convert("RGBA"), self.layer)
        w, h = comp.size
        disp = comp.resize((int(w * self.zoom), int(h * self.zoom)), Image.LANCZOS)
        self.tkimg = ImageTk.PhotoImage(disp)
        self.canvas.delete("IMG")
        self.canvas.create_image(
            self.pan_x, self.pan_y, anchor="nw", image=self.tkimg, tags="IMG"
        )
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def _record_history(self):
        self.history.append(self.layer.copy())
        if len(self.history) > 50:
            self.history.pop(0)
        self.redo_stack.clear()

    def _on_left_press(self, ev):
        self._current_stroke_id = []
        cx = self.canvas.canvasx(ev.x) - self.pan_x
        cy = self.canvas.canvasy(ev.y) - self.pan_y
        ix, iy = int(cx / self.zoom), int(cy / self.zoom)

        if not self.fill_var.get() and self.mode == "draw":
            self._current_stroke_ids = []
            self.stroke_start_x, self.stroke_start_y = ix, iy

        if self.fill_var.get():
            floodfill(self.layer, (ix, iy), self.pen_color)
            self._record_history()
            self._show_image()
            if self.on_image_change:
                self.on_image_change(self.image_paths[self.idx], self.layer)
            return
        if not self.drawing:
            self.drawing = True
            self.stroke_start_x, self.stroke_start_y = ix, iy
            self.last_x, self.last_y = ix, iy
            self._show_image()
            if self.on_image_change:
                self.on_image_change(self.image_paths[self.idx], self.layer)

    def _on_left_move(self, ev):
        if self.fill_var.get() or not self.drawing:
            return
        cx = self.canvas.canvasx(ev.x) - self.pan_x
        cy = self.canvas.canvasy(ev.y) - self.pan_y
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

    def _on_left_release(self, ev):
        if self.drawing and self.mode == "draw" and not self.fill_var.get():

            drawer = ImageDraw.Draw(self.layer)
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
            drawer.line(
                [
                    (self.last_x, self.last_y),
                    (self.stroke_start_x, self.stroke_start_y),
                ],
                fill=self.pen_color,
                width=self.pen_width,
            )
            for sid in self._current_stroke_ids:
                self.canvas.delete(sid)
            self._current_stroke_ids = []
            self._record_history()
            self._show_image()
            if self.on_image_change:
                self.on_image_change(self.image_paths[self.idx], self.layer)
        self.drawing = False

    def _on_right_press(self, ev):
        self._prx, self._pry = ev.x, ev.y

    def _on_right_move(self, ev):
        dx, dy = ev.x - self._prx, ev.y - self._pry
        self._prx, self._pry = ev.x, ev.y
        self.pan_x += dx
        self.pan_y += dy
        self._show_image()
        if self.on_image_change:
            self.on_image_change(self.image_paths[self.idx], self.layer)

    def _on_right_release(self, ev):
        self._prx = self._pry = None

    def _on_mousewheel(self, ev):
        self.zoom *= 1.1 if ev.delta > 0 else 0.9
        if self._zoom_after_id:
            self.canvas.after_cancel(self._zoom_after_id)
        self._zoom_after_id = self.canvas.after(self._apply_zoom)

    def _do_zoom_redraw(self):
        self._zoom_after_id = None
        self._show_image()

    def set_mode(self, m):
        self.mode = m
        self.canvas.config(cursor="pencil" if m == "draw" else "circle")

    def _set_zoom(self, factor):
        self.zoom = max(0.1, min(self.zoom * factor, 10.0))
        if self.on_image_change:
            self.on_image_change(self.image_paths[self.idx], self.layer)

    def clear_all(self):
        self.layer = Image.new("RGBA", self.base.size, (0, 0, 0, 0))
        self.drawer = ImageDraw.Draw(self.layer)
        self.history = [self.layer.copy()]
        self.redo_stack.clear()
        self.note_text.delete("1.0", "end")
        self.inspector_var.set("")
        for v in self.defect_vars.values():
            v.set(False)
        for v in self.pen_vars.values():
            v.set("")
        self._show_image()
        if self.on_image_change:
            self.on_image_change(self.image_paths[self.idx], self.layer)
        self._unsaved_clear = True

    def undo(self):
        if len(self.history) > 1:
            self.redo_stack.append(self.history.pop())
            self.layer = self.history[-1].copy()
            self.drawer = ImageDraw.Draw(self.layer)
            self._show_image()
            if self.on_image_change:
                self.on_image_change(self.image_paths[self.idx], self.layer)

    def redo(self):
        if self.redo_stack:
            img = self.redo_stack.pop()
            self.history.append(img.copy())
            self.layer = img.copy()
            self.drawer = ImageDraw.Draw(self.layer)
            self._show_image()
            if self.on_image_change:
                self.on_image_change(self.image_paths[self.idx], self.layer)

    def prev_image(self):
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

    def next_image(self):
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

    def _on_close(self):
        if not self._unsaved_clear:
            self._save_current()
        self.root.destroy()

    def save(self):
        logger.info(f"Saving annotations for image index {self.idx}")
        now = datetime.datetime.now()
        file_ts = now.strftime(TS_FILENAME_FMT)
        disp_ts = now.strftime(TS_DISPLAY_FMT)
        inspector = self.inspector_var.get().strip() or "unknown"
        tray = self.tray_var.get().strip() or ""
        input_folder = os.path.basename(self.input_dir.rstrip(os.sep))
        base_name = os.path.splitext(os.path.basename(self.image_paths[self.idx]))[0]
        sels = [d for d, v in self.defect_vars.items() if v.get()]
        if not sels:
            sels = ["good"]
        pen_labels = {k: self.pen_vars[k].get() for k in self.pen_vars}
        pen_to_color = {
            "pen1": PEN1_COLOR,
            "pen2": PEN2_COLOR,
            "pen3": PEN3_COLOR,
            "pen4": PEN4_COLOR,
            "pen5": PEN5_COLOR,
            "pen6": PEN6_COLOR,
        }
        COLOR_TOL = 10

        # Save individual masks, overwriting any existing ones
        for defect in sels:
            outd = os.path.join(self.output_dir, defect)
            os.makedirs(outd, exist_ok=True)
            # Remove any previous mask files for this image & defect
            pattern = os.path.join(outd, f"{input_folder}-*-{defect}-{base_name}.png")
            for fp in glob.glob(pattern):
                try:
                    os.remove(fp)
                except OSError:
                    pass

            matching_colors = []
            for pen_key, label_text in pen_labels.items():
                if label_text.strip().lower() == defect.strip().lower():
                    color_tuple = pen_to_color[pen_key]
                    if len(color_tuple) > 3:
                        color_tuple = tuple(color_tuple[:3])
                    matching_colors.append(color_tuple)

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
                            abs(r - pr) <= COLOR_TOL
                            and abs(g - pg) <= COLOR_TOL
                            and abs(b - pb) <= COLOR_TOL
                        ):
                            dst[x, y] = (pr, pg, pb, a)
                            break

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
                f"{input_folder}-{inspector}-{file_ts}-{defect}-{base_name}.png"
            )
            out_path = os.path.join(outd, out_filename)
            rgb_mask.save(out_path)

        all_defects = list(self.defect_vars.keys())
        for defect in all_defects:
            if defect not in sels:
                outd = os.path.join(self.output_dir, defect)
                pattern = os.path.join(
                    outd, f"{input_folder}-*-{defect}-{base_name}.png"
                )
                for fp in glob.glob(pattern):
                    try:
                        os.remove(fp)
                    except OSError:
                        pass

        # --- BEGIN combined‐mask overwrite logic ---
        combined_dir = os.path.join(self.output_dir, "masks")
        os.makedirs(combined_dir, exist_ok=True)

        # Build the short (placeholder) name and full long name
        short_name = f"{input_folder}-COMBINED-{base_name}.png"
        short_path = os.path.join(combined_dir, short_name)

        # On first real save, generate & remember a timestamp
        if not hasattr(self, "_combined_ts"):
            now = datetime.datetime.now()
            # Use the same format TS_FILENAME_FMT that you use elsewhere
            self._combined_ts = now.strftime(TS_FILENAME_FMT)

        all_defects = "_".join(sels) or "good"
        inspector = self.inspector_var.get().strip() or "unknown"
        long_name = (
            f"{input_folder}-{inspector}-"
            f"{self._combined_ts}-{all_defects}-COMBINED-{base_name}.png"
        )
        long_path = os.path.join(combined_dir, long_name)

        # Remove any existing combined masks (placeholder or previous) for this base image
        combined_pattern = os.path.join(
            combined_dir, f"{input_folder}*-COMBINED-{base_name}.png"
        )
        for fp in glob.glob(combined_pattern):
            try:
                os.remove(fp)
            except OSError:
                pass

        # Ensure all short-format placeholder masks for this base image are removed
        short_pattern = os.path.join(
            combined_dir, f"{input_folder}-COMBINED-{base_name}.png"
        )
        placeholder_matches = glob.glob(short_pattern)
        for placeholder in placeholder_matches:
            try:
                os.remove(placeholder)
            except OSError:
                print(f"Failed to remove placeholder: {placeholder}")

        # Save (or overwrite) the long‐named combined mask
        self.layer.save(long_path)

        # metadata update (unchanged)
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
                "pen_labels": {k: self.pen_vars[k].get() for k in self.pen_vars},
                "last_saved": disp_ts,
                "model": self.model_status,
                "confidence": self.model_confidence,
            }
        )
        meta[key] = entry
        json.dump(meta, open(meta_path, "w"), indent=2)
        self.save_ts_var.set(disp_ts)
        messagebox.showinfo("Saved", f"Masks exported to: {', '.join(sels)}")
        logger.info(f"Masks exported to: {', '.join(sels)}")

    def _save_current(self):
        logger.debug(f"Auto-saving placeholder mask for index {self.idx}")
        # Determine base filename without extension
        name = os.path.splitext(os.path.basename(self.image_paths[self.idx]))[0]
        # Get input folder name
        input_folder = os.path.basename(self.input_dir.rstrip(os.sep))
        # Prepare masks directory
        mask_dir = os.path.join(self.output_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        # Build filename: [input_folder]-COMBINED-[name].png
        mask_name = f"{input_folder}-COMBINED-{name}.png"
        mp = os.path.join(mask_dir, mask_name)
        # Save the combined mask layer
        self.layer.save(mp)

        # Update metadata without altering the last_saved timestamp
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
                "pen_labels": {k: self.pen_vars[k].get() for k in self.pen_vars},
                "last_saved": entry.get("last_saved", "never"),
            }
        )
        meta[key] = entry
        json.dump(meta, open(meta_path, "w"), indent=2)

    def export_excel(self):
        logger.info("Exporting metadata to Excel")
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
            model = e.get("model", "N/A").strip()

            # Inspector’s A/R
            if not defects and last_saved in ("never", ""):
                insp_pred = "Unlabeled"
            elif not defects:
                insp_pred = "Accept"
            else:
                insp_pred = "Reject"

            # Model’s A/R (from metadata)
            model_pred = model

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

        # Auto‐size columns
        for col in ws.columns:
            max_len = max((len(str(c.value)) for c in col), default=0)
            ws.column_dimensions[col[0].column_letter].width = max_len + 2

        wb.save(excel_path)
        messagebox.showinfo("Export Excel", f"Excel saved to:\n{excel_path}")

    def export_csv(self):
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
                if not defects and last_saved == "never" or last_saved == "":
                    status = "Unlabeled"
                elif not defects:
                    status = "Accept"
                else:
                    status = "Reject"
                writer.writerow([name, tray, insp, status, ", ".join(defects), note])
        messagebox.showinfo("Export CSV", f"CSV saved to:\n{csv_path}")

    def _fit_image_to_canvas(self):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw > 1 and ch > 1:
            bw, bh = self.base.size
            self.zoom = min(cw / bw, ch / bh, 1.0)

    def _meta_path(self):
        return os.path.join(self.output_dir, "metadata.json")

    def _load_meta(self):
        if os.path.isfile(self._meta_path()):
            logger.debug("Loading existing metadata")
            return json.load(open(self._meta_path()))
        return {}

    def _log_button(self, event):
        try:
            logger.info(
                f"Mouse button {event.num} clicked at ({event.x}, {event.y}) on {event.widget}"
            )
        except Exception:
            pass

    def _log_key(self, event):
        try:
            logger.info(f"Key {event.keysym} pressed on ({event.widget})")
        except Exception:
            pass

    def _log_resize(self, event):
        if event.widget == self.root:
            logger.info(f"Window resized to {event.width}x{event.height}")


# ─── Bundle Path Utility ────────────────────────────────────────────────────
def get_base_path():
    """
    Returns the path to bundled data files:
      - When frozen by PyInstaller: sys._MEIPASS
      - Otherwise: script directory
    """
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(__file__)


def _refresh_pen_widgets(self):
    # Hide all widgets
    for widgets in self.pen_widgets.values():
        for w in widgets:
            w.grid_remove()
    # Show widgets for selected checkboxes
    visible_keys = set()
    mapping = {
        "chip": "pen1",
        "scratch": "pen2",
        "gouge": "pen3",
        "inclusion": "pen4",
        "void": "pen5",
        "other": "pen6",
    }
    for defect, var in self.defect_vars.items():
        if var.get() and defect in mapping:
            visible_keys.add(mapping[defect])

    for i, key in enumerate(sorted(visible_keys)):
        label, combo, btn = self.pen_widgets[key]
        label.grid(row=2 * i, column=0, sticky="w", padx=5)
        combo.grid(row=2 * i, column=1, sticky="ew", padx=5)
        btn.grid(row=2 * i + 1, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 5))
    self.pen_frame.columnconfigure(1, weight=1)


class InferenceTool:

    def __init__(self, parent, model_path="best_val_acc_model.pth"):
        self.parent = parent
        self.model_path = (
            Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / model_path
        )
        self._build_ui()
        self.all_id = {}
        

    def _build_ui(self):
        # ─── Helper functions ───────────────────────────────────────────────────
        def on_scroll(e, t, canvases, zooms, ids, pil_images):
            # Mouse‐wheel zoom for one canvas
            factor = (
                1.1 if getattr(e, "delta", 0) > 0 or getattr(e, "num", 0) == 4 else 0.9
            )
            zooms[t] *= factor
            pil = pil_images[t]
            size = (int(pil.width * zooms[t]), int(pil.height * zooms[t]))
            resized = pil.resize(size, Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            c = canvases[t]
            c.photo = photo
            c.delete("all")
            ids[t] = c.create_image(0, 0, anchor="nw", image=photo)
            c.config(scrollregion=c.bbox("all"))

        def on_scroll_all(e, t):
            on_scroll(e, t, self.all_c, self.all_zoom, self.all_id, self.pil_imgs)

        def display(orig, heat, ov, summary, mask_layer=None):
            # Show arrays in each tab
            imgs = {
                "Original": Image.fromarray(orig),
                "Heatmap": Image.fromarray(heat),
                "Overlay": Image.fromarray(ov),
            }
            if mask_layer:
                base_ov = imgs["Overlay"].convert("RGBA")
                comp = Image.alpha_composite(base_ov, mask_layer)
                imgs["Mask Overlay"] = comp.convert("RGB")
            else:
                imgs["Mask Overlay"] = imgs["Overlay"].convert("RGB")

            for key, pil in imgs.items():
                c = self.canvases[key]
                cw, ch = c.winfo_width(), c.winfo_height()
                if cw > 1 and ch > 1:
                    f = min(cw / pil.width, ch / pil.height, 1.0)
                    self.zooms[key] = f
                else:
                    f = self.zooms[key]
                w, h = int(pil.width * f), int(pil.height * f)
                resized = pil.resize((w, h), Image.LANCZOS)

                self.pil_imgs[key] = pil  # keep original

                photo = ImageTk.PhotoImage(resized)
                c.photo = photo
                c.delete("all")
                self.ids[key] = c.create_image(
                    self.pan_x, self.pan_y, anchor="nw", image=photo
                )
                c.config(scrollregion=c.bbox("all"))

                if key in self.all_c:
                    ac = self.all_c[key]
                    ac.photo = photo
                    ac.delete("all")
                    ac.create_image(self.pan_x, self.pan_y, anchor="nw", image=photo)
                    ac.config(scrollregion=ac.bbox("all"))

            # update report text
            self.txt.config(state="normal")
            self.txt.delete("1.0", "end")
            self.txt.insert("1.0", summary)
            self.txt.config(state="disabled")

        def load_image():
            # Ask file, run inference, then display
            path = filedialog.askopenfilename(
                title="Select Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
            )
            if not path:
                return
            orig, heat, ov, summary = run_inference(str(self.model_path), path)
            display(orig, heat, ov, summary)

        self.select_image = load_image
        self.display = display

        # ─── File menu ───────────────────────────────────────────────────────────
        # self.menu = tk.Menu(self.parent)
        # file_menu = tk.Menu(self.menu, tearoff=0)
        # file_menu.add_command(label="Open Image", command=load_image)
        # self.menu.add_cascade(label="File", menu=file_menu)
        # try:
        #     self.parent.winfo_toplevel().configure(menu=self.menu)
        # except Exception:
        #     pass

        # ─── Tabs & Canvases ─────────────────────────────────────────────────────
        self.nb = ttk.Notebook(self.parent)
        self.nb.pack(expand=True, fill="both")

        # Single‐view tabs
        self.canvases, self.zooms, self.ids, self.pil_imgs = {}, {}, {}, {}
        for title in ("Original", "Heatmap", "Overlay"):
            frame = ttk.Frame(self.nb)
            self.nb.add(frame, text=title)
            c = tk.Canvas(frame, bg="black")
            c.pack(expand=True, fill="both")
            self.canvases[title] = c
            self.zooms[title] = 1.0

        mask_frame = ttk.Frame(self.nb)
        self.nb.add(mask_frame, text="Mask Overlay")
        mc = tk.Canvas(mask_frame, bg="black")
        mc.pack(expand=True, fill="both")
        self.canvases["Mask Overlay"] = mc
        self.zooms["Mask Overlay"] = 1.0

        # All‐views tab
        all_frame = ttk.Frame(self.nb)
        self.nb.add(all_frame, text="All Views")
        self.all_c, self.all_zoom, self.all_id = {}, {}, {}

        all_frame.grid_rowconfigure(0, weight=1)
        all_frame.grid_rowconfigure(1, weight=1)
        all_frame.grid_columnconfigure(0, weight=1)
        all_frame.grid_columnconfigure(1, weight=1)

        for i, title in enumerate(("Original", "Heatmap", "Overlay", "Mask Overlay")):
            row, col = divmod(i,2)
            c = tk.Canvas(all_frame, bg="black", highlightthickness=0)
            c.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
            self.all_c[title] = c
            self.all_zoom[title] = 1.0
            self.all_id[title] = None

        self.nb.pack(expand=1, fill="both")
        # Report tab
        rep_frame = ttk.Frame(self.nb)
        self.nb.add(rep_frame, text="Report")
        self.txt = tk.Text(rep_frame, wrap="word", state="disabled")
        self.txt.pack(expand=True, fill="both")

        self.nb.select(all_frame)

        self.pan_x = 0
        self.pan_y = 0
        self._prx = None
        self._pry = None

        def on_zoom(e):
            factor = (
                1.1 if getattr(e, "delta", 0) > 0 or getattr(e, "num", 0) == 4 else 0.9
            )
            for k in self.zooms:
                self.zooms[k] = self.zooms[k] * factor
            for k, pil in self.pil_imgs.items():
                w, h = int(pil.width * self.zooms[k]), int(pil.height * self.zooms[k])
                img_resized = pil.resize((w, h), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img_resized)
                # single‐view
                c = self.canvases[k]
                c.photo = photo
                c.delete("all")
                self.ids[k] = c.create_image(
                    self.pan_x, self.pan_y, anchor="nw", image=photo
                )
                c.config(scrollregion=c.bbox("all"))
                # all‐views (only for Orig/Heat/Overlay)
                if k in self.all_c:
                    ac = self.all_c[k]
                    ac.photo = photo
                    ac.delete("all")
                    ac.create_image(self.pan_x, self.pan_y, anchor="nw", image=photo)
                    ac.config(scrollregion=ac.bbox("all"))

        def on_pan_start(e):
            self._prx, self._pry = e.x, e.y

        def on_pan_move(e):
            dx, dy = e.x - self._prx, e.y - self._pry
            self._prx, self._pry = e.x, e.y
            self.pan_x += dx
            self.pan_y += dy

            for k, cid in self.ids.items():
                c = self.canvases[k]
                c.coords(cid, self.pan_x, self.pan_y)
                c.config(scrollregion=c.bbox("all"))
                if k in self.all_c:
                    ac = self.all_c[k]
                    ac.coords(cid, self.pan_x, self.pan_y)
                    ac.config(scrollregion=ac.bbox("all"))

        for c in list(self.canvases.values()) + list(self.all_c.values()):
            # zoom wheels
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                c.bind(ev, on_zoom)
            # pan right‐click
            for btn in (2, 3):
                c.bind(f"<Button-{btn}>", on_pan_start)
                c.bind(f"<B{btn}-Motion>", on_pan_move)
        # ─── Model lookup & initial run ───────────────────────────────────────────
        base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        self.model_path = Path(base) / self.model_path
        if not self.model_path.exists():
            messagebox.showerror("Error", f"Model not found: {self.model_path}")
            return

        # Kick off first inference
        # load_image()

        # self.frame = ttk.Frame(self.parent)
        # self.frame.pack(fill="both", expand=True, padx=10, pady=10)

        # self.label = ttk.Label(self.frame, text="Select an image for inference:")
        # self.label.pack(pady=(0, 10))

        # self.select_button = ttk.Button(
        #     self.frame, text="Select Image", command=self.select_image
        # )
        # self.select_button.pack(pady=(0, 10))

        # self.result_text = tk.Text(
        #     self.frame, height=15, width=60, wrap="word", state="disabled"
        # )
        # self.result_text.pack(pady=(10, 0))

    # Packaging:
    # Windows single-file:  pyinstaller --onefile --windowed --add-data "best_val_acc_model.pth;." gui_infer.py
    # macOS/Linux single-file: pyinstaller --onefile --add-data "best_val_acc_model.pth:." gui_infer.py

    def transforms_eval(transforms):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def overlay(cam, img_rgb, alpha=0.4):
        heat = np.uint8(255 * cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0])))
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(img_rgb, 1 - alpha, heat, alpha, 0)

    def build_resnet(arch):
        fns = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }
        if arch not in fns:
            raise ValueError(f"Unsupported arch '{arch}'")
        m = fns[arch](weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m

    def guess_arch(state_dict):
        fc_in = state_dict.get("fc.weight").shape[1]
        if fc_in == 512:
            return "resnet18"
        blocks = len(
            {
                k.split(".")[1]
                for k in state_dict
                if k.startswith("layer3.") and ".conv1.weight" in k
            }
        )
        return {6: "resnet50", 23: "resnet101", 36: "resnet152"}.get(blocks, "resnet50")

    def param_stats(m):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model.eval()
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None
            self._register_hooks()

        def _register_hooks(self):
            self.target_layer.register_forward_hook(
                lambda m, i, o: setattr(self, "activations", o.detach())
            )
            self.target_layer.register_backward_hook(
                lambda m, gi, go: setattr(self, "gradients", go[0].detach())
            )

        def __call__(self, x, class_idx=None):
            logits = self.model(x)
            if class_idx is None:
                class_idx = logits.argmax(1).item()
            self.model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward(retain_graph=True)
            w = self.gradients.mean(dim=[2, 3], keepdim=True)
            cam = torch.relu((w * self.activations).sum(1, keepdim=True))
            cam = nn.functional.interpolate(
                cam, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            cam = cam.squeeze().cpu().numpy()
            cam -= cam.min()
            cam /= cam.max().clip(1e-8)
            return cam

    def run_inference(model_path, image_path):
        dev = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            arch = ckpt.get("arch") or guess_arch(ckpt["state_dict"])
            state = ckpt["state_dict"]
        else:
            state = ckpt
            arch = guess_arch(state)
        m = build_resnet(arch).to(dev)
        m.load_state_dict(state)
        total, train = param_stats(m)

        img = Image.open(image_path).convert("RGB")
        orig = np.array(img)
        x = transforms_eval()(img).unsqueeze(0).to(dev)
        with torch.no_grad():
            logits = m(x)
            probs = torch.softmax(logits, 1).squeeze().cpu().numpy()
        idx = int(probs.argmax())
        classes = ["Accept", "Reject"]
        pred = classes[idx]
        cam = GradCAM(m, m.layer4[-1].conv2)(x, idx)
        heat = cv2.cvtColor(
            cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB,
        )
        ov = overlay(cam, orig)
        fc_in = m.fc.in_features
        blocks = [len(m.layer1), len(m.layer2), len(m.layer3), len(m.layer4)]
        summary = (
            "== Inference Results ==\n"
            f"Image              : {image_path}\n"
            f"Checkpoint         : {model_path}\n"
            f"Predicted class    : {pred}\n"
            f"Probabilities      : Accept={probs[0]:.4f} | Reject={probs[1]:.4f}\n\n"
            "== Model Details ==\n"
            f"Architecture       : {arch}\n"
            f"Total parameters   : {total:,}\n"
            f"Trainable params   : {train:,}\n"
            f"Final FC in-features: {fc_in}\n"
            f"Blocks per stage   : {blocks} (layers1-4)\n\n"
            "Inference completed successfully."
        )
        return orig, heat, ov, summary

    def main(self):
        self.root = self.parent
        self.root.title("Grad-CAM Inference Viewer")
        self.root.geometry("1000x700")
        # Menu for loading
        mb = tk.Menu(self.root)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Open Image", command=lambda: load_image())
        mb.add_cascade(label="File", menu=fm)
        self.root.config(menu=mb)
        nb = ttk.Notebook(self.root)
        nb.pack(expand=True, fill="both")
        titles = ["Original", "Heatmap", "Overlay"]
        canvases, zooms, ids, pil_imgs = {}, {}, {}, {}
        for t in titles:
            f = ttk.Frame(nb)
            nb.add(f, text=t)
            c = tk.Canvas(f, bg="black")
            c.pack(expand=True, fill="both")
            canvases[t] = c
            zooms[t] = 1.0
        # All views side-by-side
        af = ttk.Frame(nb)
        nb.add(af, text="All Views")
        all_c, all_zoom, all_id = (
            {},
            {},
            {},
        )
        for i, t in enumerate(titles):
            c = tk.Canvas(af, bg="black")
            c.grid(row=0, column=i, sticky="nsew")
            af.columnconfigure(i, weight=1)
            all_c[t] = c
            all_zoom[t] = 1.0
        # Report tab
        rf = ttk.Frame(nb)
        nb.add(rf, text="Report")
        txt = tk.Text(rf, wrap="word")
        txt.config(state="disabled")
        txt.pack(expand=True, fill="both")

        # Scroll handlers
        def on_scroll(e, t, cd, zm, idc, pd):
            f = (
                1.1
                if (getattr(e, "delta", 0) > 0 or getattr(e, "num", 0) == 4)
                else 0.9
            )
            zm[t] *= f
            sz = (int(pd[t].width * zm[t]), int(pd[t].height * zm[t]))
            r = pd[t].resize(sz, Image.LANCZOS)
            ptk = ImageTk.PhotoImage(r)
            c = cd[t]
            c.photo = ptk
            c.delete("all")
            idc[t] = c.create_image(0, 0, anchor="nw", image=ptk)
            c.config(scrollregion=c.bbox("all"))

        def on_scroll_all(e, t):
            on_scroll(e, t, all_c, all_zoom, all_id, pil_imgs)

        # Display update
        def display(orig, heat, ov, summary):
            imgs = {"Original": orig, "Heatmap": heat, "Overlay": ov}
            for t, a in imgs.items():
                pil = Image.fromarray(a)
                pil_imgs[t] = pil
                zooms[t] = 1.0
                all_zoom[t] = 1.0
                # single
                ph = ImageTk.PhotoImage(pil)
                c = canvases[t]
                c.photo = ph
                c.delete("all")
                ids[t] = c.create_image(0, 0, anchor="nw", image=ph)
                c.config(scrollregion=c.bbox("all"))
                for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    c.bind(
                        ev,
                        lambda e, t=t: on_scroll(e, t, canvases, zooms, ids, pil_imgs),
                    )
                # all
                ph2 = ImageTk.PhotoImage(pil)
                ac = all_c[t]
                ac.photo = ph2
                ac.delete("all")
                all_id[t] = ac.create_image(0, 0, anchor="nw", image=ph2)
                ac.config(scrollregion=ac.bbox("all"))
                for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    ac.bind(ev, lambda e, t=t: on_scroll_all(e, t))
            txt.config(state="normal")
            txt.delete("1.0", "end")
            txt.insert("1.0", summary)
            txt.config(state="disabled")

        # Load image
        def load_image():
            p = filedialog.askopenfilename(
                title="Select Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
            )
            if not p:
                return
            orig, heat, ov, summary = run_inference(str(model_path), p)
            display(orig, heat, ov, summary)

        self.select_image = load_image

        # Locate model
        script_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
        model_path = script_dir / "best_val_acc_model.pth"
        if not model_path.exists():
            messagebox.showerror("Error", f"Model not found: {model_path}")
            self.root.destroy()
            return
        load_image()

    def load_image_from_path(self, path, mask_layer=None):
        if not path:
            return
        orig, heat, ov, summary = run_inference(str(self.model_path), path)
        self.display(orig, heat, ov, summary, mask_layer)


class MergeApp:
    def __init__(self):
        # Top‐level window
        self.root = tk.Tk()
        self.root.title("Merged Masking + Inference Tool")
        self.root.geometry("1200x800")

        # Prepare config for masking tool
        config_file = "config.ini"
        config = configparser.ConfigParser()
        config.read(config_file)
        if "Paths" not in config:
            config["Paths"] = {}
        input_folder = config["Paths"].get("input_folder", "")
        if not input_folder or not os.path.isdir(input_folder):
            input_folder = filedialog.askdirectory(title="Select input folder")
            if not input_folder:
                messagebox.showerror("No input folder", "Need input folder. Exiting.")
                sys.exit(1)
            config["Paths"]["input_folder"] = input_folder
            with open(config_file, "w") as f:
                config.write(f)
        defects_cfg = config["Paths"].get("defects_config", "defects_config.json")
        dataset = config["Paths"].get("dataset", "bowtie")
        output_folder = os.path.join(input_folder, "output")
        os.makedirs(output_folder, exist_ok=True)

        # Notebook for tabs
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
        notebook.add(infer_frame, text="Inference Tool")
        self.infer = InferenceTool(infer_frame, model_path="best_val_acc_model.pth")

        def _on_change(path, mask_layer):
            # 1) do inference
            orig, heat, ov, summary = run_inference(str(self.infer.model_path), path)

            # 2) robustly extract status & first probability
            lines = summary.splitlines()
            # find the right lines
            pred_line = next((l for l in lines if l.startswith("Predicted class")), "")
            prob_line = next((l for l in lines if l.startswith("Probabilities")), "")
            # extract
            status = pred_line.split(":", 1)[1].strip() if ":" in pred_line else "N/A"
            confidence = "N/A"
            if ":" in prob_line:
                probs = (
                    prob_line.split(":", 1)[1].strip().split("|")[0].strip()
                )  # e.g. "Accept=0.9996"
                try:
                    val = float(probs.split("=", 1)[1])
                    confidence = f"{val*100:.1f}%"
                except Exception:
                    pass

            # 3) update the MaskingTool’s info panel with Grad‑CAM thumbnail
            self.mask.update_infer_info(status, confidence, Image.fromarray(ov))

            # 4) update the InferenceTool views (passing along the current mask)
            self.infer.display(orig, heat, ov, summary, mask_layer)

        initial = self.mask.image_paths[self.mask.idx]
        self.mask.on_image_change = _on_change
        _on_change(initial, self.mask.layer)

        try:
            self.root.winfo_toplevel().protocol("WM_DELETE_WINDOW", self.on_close)
        except Exception:
            pass
        self.root.mainloop()

    def on_close(self):
        # Call any cleanup if needed
        try:
            self.mask.app._on_close()
        except:
            pass
        try:
            self.infer._on_close()
        except:
            pass
        self.root.destroy()


if __name__ == "__main__":
    MergeApp()
