import cv2
from ultralytics import YOLO

def process_valorant_replay(video_path, enemy_model_path, head_model_path):
   
    # Load trained models (the best.pt file)
    print(f"Loading enemy model from: {enemy_model_path} and head model from: {head_model_path}")
    enemy_model = YOLO(enemy_model_path)
    head_model = YOLO(head_model_path)
    
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get the class names the models were trained on
    enemy_class_names = enemy_model.names
    head_class_names = head_model.names

    print("Processing video... Press 'q' to stop.")
    
    # Loop through the video frame by frame
    while True:
        # ret is a boolean that is True if the frame was read correctly
        ret, frame = cap.read()
        
        # If ret is False, we've reached the end of the video
        if not ret:
            print("End of video reached.")
            break
        # -------------------------------
        # Draw Crosshair Bounding Box
        # -------------------------------
        height, width, _ = frame.shape

        center_x = (width // 2) - 1  # Compensate for crosshair being offset by 1 pixel
        center_y = (height // 2) - 1

        box_size = 3

        x1 = center_x - box_size
        y1 = center_y - box_size
        x2 = center_x + box_size
        y2 = center_y + box_size

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame,"crosshair", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # -------------------------------
        # YOLO inference
        # -------------------------------

        # Run inference (detection) on the current frame
        enemy_results = enemy_model(frame, conf=0.5, verbose=False)
        
        # Process the results and draw boxes
        # The 'enemy_results' object contains all the bounding box coordinates for the enemy model
        for r in enemy_results:
            boxes = r.boxes
            
            for box in boxes:
                # Get the coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = frame[y1:y2, x1:x2] # Crop the detected enemy region for head detection
                
                # Get the confidence score and class ID
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = enemy_class_names[cls_id]
                
                # Assign colors based on the class (BGR format for OpenCV)
                if class_name == 'enemy':
                    color = (0, 165, 255) # Orange
                elif class_name == 'crosshair':
                    color = (0, 255, 0) # Green
                else:
                    color = (255, 255, 255) # White fallback
                
                # Draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw the label above the rectangle
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                #Cropping the detected enemy region for head detection
                
                head_results = head_model(cropped, conf=0.3, imgsz= 320, verbose=False)

                for hr in head_results:
                    for head_box in hr.boxes:
                        hx1, hy1, hx2, hy2 = map(int, head_box.xyxy[0])
                       
                        # Convert crop coordinates back to frame coordinates
                        hx1 += x1
                        hx2 += x1
                        hy1 += y1
                        hy2 += y1

                        cv2.rectangle(
                            frame,
                            (hx1, hy1),
                            (hx2, hy2),
                            (0, 0, 255),  # Red
                            2
                        )

                        cv2.putText(
                            frame,
                            "head",
                            (hx1, hy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2
                        )

        # Display the frame on screen
        cv2.imshow('Valorant AI Coach - Vision Test', frame)
        
        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Replace these paths with your actual file locations
    MY_VIDEO = "input/test-clip-1.mp4"
    
    MY_ENEMY_MODEL = "runs/detect/valorant_coach/enemy_model_v1/weights/best.pt"
    MY_HEAD_MODEL = "runs/detect/valorant_coach/head_model_v1/weights/best.pt"
    
    process_valorant_replay(MY_VIDEO, MY_ENEMY_MODEL, MY_HEAD_MODEL)