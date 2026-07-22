import json
from pathlib import Path

def fix_notebook(nb_path: Path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            # Check if this is the extract dataset cell
            if any("Extract Dataset" in line for line in cell["source"]):
                cell["source"] = [
                    "# ── Cell 3: Extract Dataset ─────────────────────────────────────────────────\n",
                    "import os\n",
                    "import zipfile\n",
                    "import shutil\n",
                    "\n",
                    "DRIVE_ZIP_PATH = '/content/drive/MyDrive/vision_yolo_dataset.zip'\n",
                    "EXTRACT_DIR = '/content'\n",
                    "YOLO_DIR = '/content/yolo_format'\n",
                    "\n",
                    "if os.path.exists(YOLO_DIR):\n",
                    "    shutil.rmtree(YOLO_DIR)\n",
                    "\n",
                    "if os.path.exists(DRIVE_ZIP_PATH):\n",
                    "    print(f'Extracting {DRIVE_ZIP_PATH} ...')\n",
                    "    with zipfile.ZipFile(DRIVE_ZIP_PATH, 'r') as z:\n",
                    "        z.extractall(EXTRACT_DIR)\n",
                    "    print('Extraction complete.')\n",
                    "else:\n",
                    "    print(f'[ERROR] Zip not found at {DRIVE_ZIP_PATH}')\n",
                    "\n",
                    "# Verify dataset.yaml\n",
                    "yaml_path = os.path.join(YOLO_DIR, 'dataset.yaml')\n",
                    "if os.path.exists(yaml_path):\n",
                    "    print('dataset.yaml found!')\n",
                    "    with open(yaml_path, 'r') as f:\n",
                    "        print(f.read())\n",
                    "else:\n",
                    "    print('[ERROR] dataset.yaml missing!')\n"
                ]
                
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Fixed {nb_path}")

def main():
    fix_notebook(Path("D:/VISION/notebooks/vision_engine_yolov11_train.ipynb"))
    fix_notebook(Path("D:/VISION/notebooks/vision_engine_rtdetr_train.ipynb"))

if __name__ == "__main__":
    main()
