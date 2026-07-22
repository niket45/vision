import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.detector.yolo import YoloDetector
from src.ocr.easyocr_backend import EasyOcrBackend

def test_pipeline():
    # Setup paths
    weights_path = Path("weights/vision_yolov11_baseline.pt")
    
    # We will test on an Eleceed image first, which is Korean.
    test_image = Path("datasets/raw/eleceed/eleceed 402/raw_109.jpg")
    output_path = Path(r"C:\Users\niket\.gemini\antigravity\brain\98b40860-c4c1-467f-9cb6-d29076c38e49\pipeline_test.jpg")
    
    print(f"Loading YOLOv11 from {weights_path}...")
    detector = YoloDetector(weights_path, device="cpu")
    
    print("Loading EasyOCR (Korean)... This may download models on the first run.")
    ocr = EasyOcrBackend(langs=["ko", "en"], gpu=False)
    
    print(f"\nRunning Detection on {test_image}...")
    result = detector.detect(test_image, confidence_threshold=0.25)
    
    bubbles = result.bubbles()
    print(f"Found {len(bubbles)} bubbles. Running OCR on them...")
    
    # Draw boxes and text
    img = Image.open(test_image).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Load a font (we'll just use default for now, which may not render Korean perfectly in PIL,
    # but we will print it to the console to verify accuracy).
    try:
        # Try to load a standard font that supports Korean on Windows
        font = ImageFont.truetype("malgun.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    
    total_ocr_time = 0.0
    for bubble in bubbles:
        start_time = time.time()
        ocr_result = ocr.recognize_region(test_image, bubble.bbox)
        ocr_time = (time.time() - start_time) * 1000
        total_ocr_time += ocr_time
        
        print(f" - Bubble {bubble.id}: {ocr_result.text} (conf: {ocr_result.confidence:.2f}, time: {ocr_time:.1f}ms)")
        
        # Draw on image
        x_min, y_min, x_max, y_max = bubble.bbox
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_max + 10, y_min), ocr_result.text, fill="red", font=font)
        
    print(f"\nTotal OCR Time: {total_ocr_time:.1f}ms")
        
    img.save(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    test_pipeline()
