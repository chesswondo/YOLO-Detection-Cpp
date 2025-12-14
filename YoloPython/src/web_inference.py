import cv2
import time
import numpy as np
import logging
import threading
from ultralytics import YOLO

logging.basicConfig(
    filename="web_fps_report.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

class CameraStream:
    """
    Class for async frame reading.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        
    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.stream.isOpened():
                self.stop()
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        # Always take the last frame
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


def main():
    model = YOLO("models/yolov8n.pt")

    # Warm up GPU to keep the statistics clear
    print("Warming up GPU...")
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy_frame, verbose=False)
    print("Warmup complete.")

    cam = CameraStream(src=0).start()
    time.sleep(1.0) # Camera warmup

    print("System active. Press CTRL+C to stop.")

    prev_time = time.time()
    frame_count = 0
    fps_display = 0.0

    # Statistics
    inference_times = []
    frame_times = []

    try:
        while True:
            t_frame_start = time.perf_counter()

            frame = cam.read()
            if frame is None: break

            # Inference measurement
            start_infer = time.perf_counter()
            results = model(frame, verbose=False, conf=0.5)
            end_infer = time.perf_counter()
            
            inf_time = (end_infer - start_infer) * 1000
            inference_times.append(inf_time)

            annotated_frame = results[0].plot()

            # Update FPS
            frame_count += 1
            curr_time = time.time()
            elapsed = curr_time - prev_time
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                prev_time = curr_time

            info_text = f"FPS: {int(fps_display)} | Inf: {int(inf_time)} ms"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Yolo Python Demo", annotated_frame)

            if cv2.waitKey(1) == 27:
                break

            t_frame_end = time.perf_counter()
            frame_times.append((t_frame_end - t_frame_start) * 1000)

    except KeyboardInterrupt:
        pass

    finally:
        cam.stop()
        cv2.destroyAllWindows()

        # Final report
        if inference_times:
            inf_arr = np.array(inference_times)
            frame_arr = np.array(frame_times)

            total_frames = len(inf_arr)
            total_time_sec = np.sum(frame_arr) / 1000

            avg_inf = np.mean(inf_arr)
            std_inf = np.std(inf_arr)
            
            avg_frame_time = np.mean(frame_arr)
            std_frame_time = np.std(frame_arr)
            
            real_fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
            potential_fps = 1000 / avg_inf if avg_inf > 0 else 0

            report = (
                f"\nPERFORMANCE REPORT:\n"
                f"Total Frames Processed: {total_frames}\n"
                f"Total Duration:         {total_time_sec:.2f} sec\n"
                f"\n"
                f"INFERENCE STATS (CV model only):\n"
                f"  Avg Time: {avg_inf:.2f} ms\n"
                f"  Std Dev:  {std_inf:.2f} ms (Jitter)\n"
                f"  Max Potential FPS: {potential_fps:.2f}\n"
                f"\n"
                f"SYSTEM STATS (end-to-end):\n"
                f"  Avg Frame Time: {avg_frame_time:.2f} ms\n"
                f"  Std Dev:        {std_frame_time:.2f} ms\n"
                f"  Real Average FPS: {real_fps:.2f}\n"
            )
            
            print(report)
            logging.info(report)
        else:
            logging.warning("No frames processed.")


if __name__ == "__main__":
    main()
