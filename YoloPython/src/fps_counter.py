import time
import cv2
import numpy as np
import requests
import logging
from ultralytics import YOLO

logging.basicConfig(
    filename="test_fps_report.txt",
    filemode="w",
    level=logging.INFO,
    format="%(message)s"
)

def download_test_image(img_url: str):
    img_data = requests.get(img_url).content
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def benchmark_python(model, image, runs=200):
    logging.info("Warming up GPU...")
    for _ in range(10):
        model(image, verbose=False)
    
    logging.info(f"Running {runs} iterations...")
    
    start_time = time.perf_counter()
    
    for _ in range(runs):
        results = model(image, verbose=False)
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / runs) * 1000
    fps = runs / total_time
    
    logging.info(f"Total time: {total_time:.2f} seconds")
    logging.info(f"Average time per frame: {avg_time_ms:.2f} ms")
    logging.info(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    img_url = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"
    img = download_test_image(img_url)
    if img is None:
        logging.error("Error downloading image")
        exit()
    
    model = YOLO("models/yolov8n.pt")
    benchmark_python(model, img)