from ultralytics import YOLO

def train_head_model():
    # Load pre-trained YOLO11 Nano model
    print("Loading pre-trained YOLO model...")
    model = YOLO("yolo11n.pt")

    print("Starting head detection training...")

    results = model.train(
        # Path to your head detection dataset
        data=r"C:\Users\dhira\Projects\valorant-aim-analyzer\data\head_dataset\data.yaml",

        # Training
        epochs=100,
        patience=20,
        imgsz=960,          # Larger image size helps detect small heads
        batch=8,            # Reduce if you run out of GPU memory
        device=0,
        workers=0,

        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.4,
        fliplr=0.5,
        mosaic=1.0,

        # Save location
        project="valorant_coach",
        name="head_model_v1"
    )

    print("Training complete! The best model weights are saved in 'valorant_coach/head_model_v1/weights/best.pt'")


if __name__ == "__main__":
    train_head_model()