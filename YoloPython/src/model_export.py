from ultralytics import YOLO

def export_model():
    model = YOLO("models/yolov8n.pt")
    model.export(
        format="onnx",
        dynamic=False,
        opset=12,
        nms=True,
        project="models",
        name="yolov8n"
    )

    return model

if __name__ == "__main__":
    model = export_model()