# AnnoMate - The Anomaly Detection Masking Tool

A Tkinter-based GUI tool to load images from an input directory, draw/erase colored “masks” (for different defect categories), and export those masks (per category) to separate output folders. It also lets you save metadata (inspector name, tray/directory, notes, defects, pen labels) and export a CSV or Excel sheet summarizing all annotations.

## Table of Contents

- [Project Overview](#project-overview)
- [Branch Variants](#branch-variants)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Exported Outputs](#exported-outputs)
- [Running from VS Code or the command line](#running-from-vs-code-or-the-command-line)
- [PyInstaller Packaging Instructions](#pyinstaller-packaging-instructions)
- [Troubleshooting](#troubleshooting)
- [Contact](#contact)

## Project Overview

This tool lets an inspector:

- Browse through a folder of images
- Mark defects using up to five different “pens” (each pen corresponds to a specific color)
- Optionally fill an entire region with a pen color (flood fill) or do freehand marks/erasing
- Assign each pen to one of the defect categories via `defects_config.json`
- Enter metadata per image (e.g., inspector name, tray/directory, free-form note)
- Record which defect categories were selected for each image

Exported Outputs include:

- **Masks**: Separate PNG mask per defect category
- **Metadata JSON**: `metadata.json` storing per-image metadata
- **Excel Sheet**: Summarizes images with columns: Filename, Tray/Directory, Inspector, Accept/Reject, Defects, Notes
- **CSV File**: Same content as Excel for easier ingestion

In the full branches, additional features:

- **Inference**: Classify images as Accept or Reject using a pretrained ResNet
- **Grad-CAM**: Display heatmap overlay to visualize influential regions

## Branch Variants

| Branch                   | OS      | Script               | Purpose/Feature                     |
|--------------------------|---------|----------------------|-------------------------------------|
| `windows-full`           | Windows | `draw_app_both.py`   | Full tool (masking + inference)     |
| `windows-masking-only`   | Windows | `draw_app.py`        | Masking only                        |
| `mac-full`               | macOS   | `draw_app_both.py`   | Full tool (masking + inference)     |
| `mac-masking-only`       | macOS   | `draw_app.py`        | Masking only                        |

## Prerequisites

- **Python 3.8+**
  - Verify: `python3 --version`
- **Tkinter**
  - Included with standard Python installer on Windows/macOS
  - On Debian/Ubuntu: `sudo apt-get install python3-tk`
- **pip**
  - Verify: `pip --version`
- **Required Packages** (in `requirements.txt`):
  - `Pillow`
  - `openpyxl`
  - Install with:
    ```bash
    pip install -r requirements.txt
    ```

### Optional Virtual Environment

```bash
python3 -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Installation

1. Clone or copy the repository.
2. Ensure you have the appropriate script (`draw_app.py` or `draw_app_both.py`), `defects_config.json`, and `requirements.txt`.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The tool reads `defects_config.json` mapping dataset names to defect category lists:

```json
{
  "default": ["scratch", "dent", "crack", "..."]
}
```

Each key maps to a dataset, and values are arrays of defect category strings displayed as checkboxes.

## Usage

### Launching the Application

```bash
# Masking-only
python draw_app.py

# Full (masking + inference)
python draw_app_both.py
```

### GUI Overview

- **Left Panel (Canvas)**
  - Display images and mask layers
  - Draw with left-click, pan with right-click
  - Zoom controls: `-` / `+`
  - Draw/Erase toggle and pen fill mode

- **Right Panel (Controls)**
  - **Navigation**: Prev / Next, Clear, Undo / Redo, Save
  - **Zoom**: `-` / `+`
  - **Information**: Note, Inspector, Tray/Directory, Last saved
  - **Categories**: Checkboxes for defects
  - **Pen Controls**: Select pen (PEN1–PEN5) and assign defect category
  - **Export**: 
    - Export as XLSX: `<input_dir>.xlsx` in output directory
    - Export as CSV: `<input_dir>.csv` in output directory

## File Structure

```
your_project_folder/
├── draw_app.py or draw_app_both.py
├── defects_config.json
├── requirements.txt
├── your_input_directory/
│   ├── image1.png
│   └── ...
└── your_output_directory/
    ├── masks/
    ├── <defect_name>/
    ├── metadata.json
    └── <input_dir>.xlsx/.csv
```

## Exported Outputs

On saving an image:

1. **Combined Mask**: `<output_dir>/masks/<image_basename>_mask.png`
2. **Per-Defect Masks**:
   - `<output_dir>/<defect_name>/YYYYMMDD_HHMMSS_<inspector>_<image_basename>.png`
3. **Metadata File**: `metadata.json` in output directory
4. **Excel/CSV Export**: `<input_dir>.xlsx` and `<input_dir>.csv` in output directory

## Running from VS Code or the command line

1. Open the project folder in VS Code.
2. (Optional) Create and activate a virtual environment.
3. Open a new terminal.
4. Run the appropriate script:
   ```bash
   python draw_app.py    # Masking-only
   python draw_app_both.py  # Full (masking + inference)
   ```

## PyInstaller Packaging Instructions

1. **Install PyInstaller**:
   ```bash
   pip install pyinstaller
   ```
2. **Basic Build**:
   ```bash
   pyinstaller draw_app.py
   ```
3. **One-file GUI Build**:
   ```bash
   pyinstaller --onefile --windowed draw_app.py
   ```
4. **Include Data Files**:
   ```bash
   # Windows
   pyinstaller --onefile --windowed --add-data "defects_config.json;." --name AnnoMateMask draw_app.py

   # macOS/Linux
   pyinstaller --onefile --windowed --add-data "defects_config.json:." --name AnnoMateMask draw_app.py
   ```
5. **Apple Silicon (optional)**:
   ```bash
   pyinstaller --onefile --windowed --target-arch universal2 --add-data "defects_config.json:." --name AnnoMateFull draw_app_both.py
   ```

## Troubleshooting

- **No module named Tkinter**: Install OS package: `sudo apt-get install python3-tk`
- **No module named PIL**: `pip install Pillow`
- **Openpyxl errors**: `pip install openpyxl`
- **JSON config errors**: Validate `defects_config.json` syntax and ensure a `default` key exists
- **Permission errors**: Choose an output directory with write access

## Contact

- **GitHub Issues**: https://github.com/cjgeo22/AD_masking_tool/issues
- **Email**: lgeorge@coastal.edu
- **Slack**: @CJ George
