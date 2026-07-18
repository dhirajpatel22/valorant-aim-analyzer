import cv2
from ultralytics import YOLO

def draw_crosshair(frame):
    """Draws a crosshair at the center of the given frame. Modifies the frame in place. Returns the center coordinates as a tuple."""
    height, width, _ = frame.shape

    center_x = (width // 2) - 1  # Compensate for crosshair being offset by 1 pixel
    center_y = (height // 2) - 1 # + 17 # temp mac adjustment

    box_size = 3

    x1 = center_x - box_size
    y1 = center_y - box_size
    x2 = center_x + box_size
    y2 = center_y + box_size

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame,"crosshair", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.circle(frame, (center_x, center_y), 1, (0, 255, 0), -1) #Crosshair center

    return (center_x, center_y) 

def draw_estimate_head(frame, enemy_box):
    """Estimates the head bounding box based on the enemy bounding box. Modifies the frame in place. Returns the estimated head center coordinates as a tuple and the estimated head box coordinates as a tuple."""
    x1, y1, x2, y2 = enemy_box
    enemy_width = x2 - x1
    enemy_height = y2 - y1

    # Estimated head center
    head_center_x = x1 + enemy_width // 2
    head_center_y = y1 + int(enemy_height * 0.13)

    # Estimated head box size
    head_box_size = int(enemy_width * 0.30)

    hx1 = head_center_x - head_box_size // 2
    hy1 = head_center_y - head_box_size // 2
    hx2 = head_center_x + head_box_size // 2
    hy2 = head_center_y + head_box_size // 2
    
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 255), 2)
    cv2.putText(frame, "head (ESTIMATE)", (hx1, hy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    return (head_center_x, head_center_y), (hx1, hy1, hx2, hy2)  # Return the estimated head center coordinates and box coordinates

def draw_detection(frame, box, class_names):
    """Draws a YOLO detection bounding box and returns head information."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # Get the confidence score and class ID
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])
    class_name = class_names[cls_id]
    
    # Assign colors based on the class (BGR format for OpenCV)
    
    head_center_x, head_center_y = None, None
    if class_name == 'enemy':
        color = (0, 165, 255) # Orange
    elif class_name == 'head':
        color = (255, 0, 255) # Magenta
        head_center_x = (x1 + x2) // 2
        head_center_y = (y1 + y2) // 2
    else:
        color = (255, 255, 255) # White fallback
    
    # Draw the rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw the label above the rectangle
    label = f"{class_name} {conf:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return (head_center_x, head_center_y), (x1, y1, x2, y2)  # Return the head center coordinates and box coordinates


def display_vertical_crosshair_error(frame, head_center_y, crosshair_x, crosshair_y, hy1, hy2):
    """Calculates the vertical crosshair error and displays it on the frame. Displays aim feedback on frame. Modifies the frame in place. Returns nothing."""
    vertical_crosshair_error = head_center_y - crosshair_y # In OpenCV, (0,0) is at the top-left corner, 
                                                           # so a positive value means the head is below the crosshair
    if hy1 <= crosshair_y <= hy2:
        cv2.putText(frame, f"Vertical Crosshair Error: {vertical_crosshair_error}px", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
    else:
        cv2.putText(frame, 
                    f"Vertical Crosshair Error: {vertical_crosshair_error}px (OUT OF BOUNDS)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
    # Create transparent overlay for arrow
    overlay = frame.copy()

    # Put arrow to the right of crosshair
    arrow_x = crosshair_x + 40
    arrow_length = 80  # cap arrow size


    if hy1 <= crosshair_y <= hy2:
        cv2.putText(overlay, "GOOD", 
                    (arrow_x, 
                    crosshair_y + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0),
                    2)
    elif crosshair_y < hy1 and vertical_crosshair_error > 0:
        # Crosshair too high
        cv2.putText(
                    overlay,
                    "TOO HIGH",
                    (crosshair_x + 40, crosshair_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                    )

    elif crosshair_y > hy2 and vertical_crosshair_error < 0:
         # Crosshair too low
        cv2.putText(
                    overlay,
                    "TOO LOW",
                    (crosshair_x + 40, crosshair_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                    )

    # Blend the overlay
    transparency = 0.6
    cv2.addWeighted(overlay, transparency, frame, 1-transparency, 0, frame)
        
def process_valorant_replay(video_path, model_path):
   
    # Load trained models (the best.pt file)
    print(f"Loading enemy and head model from: {model_path}")
    model = YOLO(model_path)
    
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get the class names the models were trained on
    class_names = model.names

    print("Processing video... Press 'q' to stop.")
    
    # Loop through the video frame by frame
    while True:
        # ret is a boolean that is True if the frame was read correctly
        ret, frame = cap.read()
        
        # If ret is False, we've reached the end of the video
        if not ret:
            print("End of video reached.")
            break
        
        (crosshair_x, crosshair_y) = draw_crosshair(frame)

        # Run inference (detection) on the current frame
        results = model(frame, conf=0.5, verbose=False)
        
        # Process the results and draw boxes
        # The 'results' object contains all the bounding box coordinates for the model
        for r in results:
            boxes = r.boxes
            
            closest_head = None
            closest_head_box = None
            closest_distance = float('inf')

            for box in boxes:
                # Get the coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cls_id = int(box.cls[0])
                class_name = class_names[cls_id]

                (head_center_x, head_center_y), (hx1, hy1, hx2, hy2) = draw_detection(frame, box, class_names)

                if class_name == 'head':
                    #Find closest head to crosshair
                    distance = ((head_center_x - crosshair_x)**2 + 
                    (head_center_y - crosshair_y)**2) ** 0.5
                    
                    #Keep the closest head
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_head = (head_center_x, head_center_y)
                        closest_head_box = (hx1, hy1, hx2, hy2)

                if closest_head is not None:
                    head_center_x, head_center_y = closest_head
                    hx1, hy1, hx2, hy2 = closest_head_box

                    # Calculate the vertical crosshair error & display it on the frame
                    display_vertical_crosshair_error(frame, head_center_y, crosshair_x, crosshair_y, hy1, hy2)         

        # Display the frame on screen
        cv2.imshow('Valorant AI Coach - Vision Test', frame)
        
        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    MY_VIDEO = "input/test-clip-3.mp4"
    MY_MODEL = "runs/detect/valorant_coach/enemy_and_head_model_v1/weights/best.pt"
 
    
    process_valorant_replay(MY_VIDEO, MY_MODEL)