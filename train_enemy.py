from ultralytics import YOLO

def train_enemy_detection_model():
    # 1. Load a pre-trained Nano model. 
    # Starting with a pre-trained model (transfer learning) is much 
    # faster and more accurate than training from scratch.
    print("Loading pre-trained YOLO model...")
    model = YOLO('yolo11n.pt')  #version of YOLO model

    # 2. Start the training process.
    print("Starting training cycle...")
    results = model.train(
        data='C:\\Users\\dhira\\Projects\\valorant-aim-analyzer\\data\\val_images_13k\\data.yaml',  # Path to your dataset YAML file
        
        # Hyperparameters
        epochs=100,      # How many times the AI looks at the entire dataset
        patience=20,     # Stop training if no improvement after 20 epochs
        imgsz=640,       # Resize all images to 640x640 for consistency
        batch=16,        # Number of images processed at once (lower this if your GPU runs out of memory)
        device=0,        # '0' uses your primary GPU. Use 'cpu' if you don't have a dedicated GPU.
        workers = 0,     # Number of CPU threads to use for data loading
        
        # Data Augmentation (Crucial for gaming environments)
        # This slightly alters the images to prevent the AI from just memorizing them.
        hsv_h=0.015,     # Slight hue shifts (helps handle different map lighting)
        hsv_s=0.4,       # Saturation shifts (helps handle Viper walls or Reyna blinds)
        fliplr=0.5,      # Flips the image left/right 50% of the time
        mosaic=1.0,      # Combines 4 images into one (helps with small object detection)
        
        project='valorant_coach', # Folder name where results are saved
        name='enemy_model_v1'       # Sub-folder for this specific training run
    )
    
    print("Training complete! The best model weights are saved in 'valorant_coach/enemy_model_v1/weights/best.pt'")

if __name__ == '__main__':
    # We put the training logic inside this block to prevent multiprocessing errors 
    # that can sometimes happen on Windows machines when training models.
    train_enemy_detection_model()