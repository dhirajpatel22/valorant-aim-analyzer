import cv2
from ultralytics import YOLO

def process_valorant_replay(video_path, model_path):
    # 1. Load your custom trained model (the best.pt file)
    print(f"Loading custom model from: {model_path}")
    model = YOLO(model_path)
    
    # 2. Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get the class names your model was trained on (e.g., 0: 'enemy_body', 1: 'enemy_head', 2: 'crosshair')
    class_names = model.names

    print("Processing video... Press 'q' to stop.")
    
    # 3. Loop through the video frame by frame
    while True:
        # ret is a boolean that is True if the frame was read correctly
        ret, frame = cap.read()
        
        # If ret is False, we've reached the end of the video
        if not ret:
            print("End of video reached.")
            break
            
        # 4. Run inference (detection) on the current frame
        # conf=0.5 means only show detections the AI is at least 50% sure about
        results = model(frame, conf=0.5, verbose=False)
        
        # 5. Process the results and draw boxes
        # The 'results' object contains all the bounding box coordinates
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                # Get the coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get the confidence score and class ID
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = class_names[cls_id]
                
                # Assign colors based on the class (BGR format for OpenCV)
                if class_name == 'enemy_head':
                    color = (0, 0, 255) # Red
                elif class_name == 'enemy_body':
                    color = (0, 165, 255) # Orange
                elif class_name == 'crosshair':
                    color = (0, 255, 0) # Green
                else:
                    color = (255, 255, 255) # White fallback
                
                # Draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw the label above the rectangle
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 6. Display the frame on screen
        cv2.imshow('Valorant AI Coach - Vision Test', frame)
        
        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Replace these paths with your actual file locations
    MY_VIDEO = "C:\\Users\\dhira\\Projects\\valorant-aim-analyzer\\data\\test-clip-1.mp4"
    MY_MODEL = "runs/detect/valorant_coach/aim_model_v1/weights/best.pt"
    
    process_valorant_replay(MY_VIDEO, MY_MODEL)