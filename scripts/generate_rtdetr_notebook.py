import json
from pathlib import Path

def main():
    yolo_nb = Path("D:/VISION/notebooks/vision_engine_yolov11_train.ipynb")
    out_nb = Path("D:/VISION/notebooks/vision_engine_rtdetr_train.ipynb")
    
    with open(yolo_nb, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "markdown":
            new_source = []
            for line in cell["source"]:
                line = line.replace("YOLOv11 Nano", "RT-DETR Large")
                line = line.replace("yolo11n", "rtdetr-l")
                line = line.replace("YOLO", "RT-DETR")
                new_source.append(line)
            cell["source"] = new_source
            
        elif cell["cell_type"] == "code":
            new_source = []
            for line in cell["source"]:
                line = line.replace("YOLOv11 Nano", "RT-DETR Large")
                line = line.replace("yolo11n.pt", "rtdetr-l.pt")
                line = line.replace("from ultralytics import YOLO", "from ultralytics import RTDETR")
                line = line.replace("model = YOLO", "model = RTDETR")
                line = line.replace("vision_v1_baseline", "vision_rtdetr_baseline")
                line = line.replace("vision_yolov11_baseline.pt", "vision_rtdetr_baseline.pt")
                
                # RT-DETR uses more memory, reduce batch size
                line = line.replace("batch=16", "batch=8")
                new_source.append(line)
            cell["source"] = new_source
            
    with open(out_nb, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
        
    print(f"Generated {out_nb}")

if __name__ == "__main__":
    main()
