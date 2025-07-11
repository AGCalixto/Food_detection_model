from ultralytics import YOLO

def main():
    # Carga el modelo base
    model = YOLO('yolov8n.pt')

    # Entrena el modelo con tu dataset y parámetros
    model.train(
        data='food101.yaml',
        epochs=10,
        imgsz=416,
        batch=32,
        device=0,
        name='food101_yolov8'
    )

if __name__ == "__main__":
    main()
