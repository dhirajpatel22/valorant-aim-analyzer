import cv2
from sympy import fps
from ultralytics import YOLO
import easyocr
from difflib import SequenceMatcher 

from dataclasses import dataclass, field
from typing import List
import re
from difflib import SequenceMatcher
from colorama import Fore, Style, init
from itertools import count

init(autoreset=True)  # Automatically reset color after each print
reader = easyocr.Reader(['en'], gpu=True)
candidate_id_counter = count()  # Global counter for unique IDs

@dataclass
class KillFeedDetection:
    """Represents a single OCR detection in the kill feed."""
    text: str
    conf: float
    x: int
    y: int
    frame_idx: int 

@dataclass
class KillFeedRow:
    """Represents a row of detections in the kill feed."""
    text: list
    y: int
    x: int
    frame_idx: int
    parts: list = field(default_factory=list)

    def __repr__(self):
        return (
            f"KillFeedRow | "
            f"text={self.text} | "
            f"pos=({self.x}, {self.y}) | "
            f"frame={self.frame_idx} | "
            f"parts={len(self.parts)}"
        )
     
@dataclass
class KillFeed:
    """Represents the entire kill feed, which consists of multiple rows."""
    rows: List[KillFeedRow] = field(default_factory=list)

    def __post_init__(self):
        self.rows.sort(key=lambda r: r.y)  # Ensure rows are sorted by their y-coordinate

    def __str__(self):
        if not self.rows:
            return "KillFeed: <empty>"

        lines = ["=" * 60, "KILL FEED", "=" * 60]

        for i, row in enumerate(self.rows):
            text = " | ".join(
                getattr(t, "text", str(t)) for t in row.text
            )

            lines.append(
                f"Row {i + 1}:"
                f"  y={row.y:<4}"
                #f"  frame={row.frame_idx:<6}"
                f"  text=[{text}]"
            )

        lines.append("=" * 60)

        return "\n".join(lines)

    def add_row(self, new_row: KillFeedRow):
        self.rows.append(new_row)
        self.rows.sort(key=lambda r: r.y)  # Keep rows sorted by their y-coordinate

    def remove_old_rows(self, current_frame: int, fps: float = 30.0, max_age: float = 5.0):
        for row in self.rows:
            current_time = current_frame / fps
            timestamp = row.frame_idx / fps
            if current_time - timestamp > max_age:
                self.rows.remove(row)

    def get_latest_row(self):
        if self.rows:
            return max(self.rows, key=lambda row: row.frame_idx)
        return None

    def clear(self):
        self.rows.clear()

@dataclass
class KillCandidate:
    """Represents a potential user kill event detected in the kill feed."""
    rows: list[KillFeedRow]
    x: int
    y: int
    first_frame: int
    last_frame: int
    ID: int = field(default_factory=lambda: next(candidate_id_counter))  # Unique identifier for the kill candidate

    def __repr__(self):
        return (
            f"KillCandidate(ID = {self.ID}) | text = {self.rows[-1].text} | frames=({self.first_frame}-{self.last_frame}) | "
            f"y={self.y}"
        )

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

def draw_enemy(frame, box, enemy_class_names):
    """Draws a bounding box around the detected enemy. Does not return anything; modifies the frame in place."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # Get the confidence score and class ID
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])
    class_name = enemy_class_names[cls_id]
    
    # Assign colors based on the class (BGR format for OpenCV)
    if class_name == 'enemy':
        color = (0, 165, 255) # Orange
    else:
        color = (255, 255, 255) # White fallback
    
    # Draw the rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw the label above the rectangle
    label = f"{class_name} {conf:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_head(frame, head_box, x1, y1, x2, y2):
    """Draws a bounding box around the detected head. Modifies the frame in place. Returns the center coordinates of the head bounding box as a tuple and the box coordinates as a tuple."""
    hx1, hy1, hx2, hy2 = map(int, head_box.xyxy[0])
                       
   # Convert crop coordinates back to frame coordinates
    hx1 += x1
    hx2 += x1
    hy1 += y1
    hy2 += y1

    box_coordinates = (hx1, hy1, hx2, hy2)

    center_x = (hx1 + hx2) // 2
    center_y = (hy1 + hy2) // 2
    center = (center_x, center_y)

    # Draw a small circle at the center of the head bounding box
    cv2.circle(frame, center, 2, (0, 0, 255), -1)

    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)
    cv2.putText(frame, "head", (hx1, hy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),2)

    return center, box_coordinates

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

def crop_kill_frame(frame):
    """ Crops the frame to focus on the kill feed area. Returns the cropped frame and the top-left coordinates of the crop in the original frame. """
    h, w, _ = frame.shape
    x1 = int(w * 0.78)
    y1 = int(h * 0.08)
    x2 = w - 20
    y2 = int(h * 0.35)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Draw rectangle on original frame for visualization

    return frame[y1:y2, x1:x2], (x1, y1)

def preprocess_kill_feed(crop):
    """Preprocesses the cropped kill feed for OCR. Returns a binary image suitable for OCR."""
    return cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)  # Resize to double the size for better OCR accuracy

def ocr_kill_feed(frame, frame_idx):
    """Performs OCR on the kill feed area of the frame. Returns a list of KillFeedDetection objects."""
    crop, (ox, oy) = crop_kill_frame(frame)
    if crop.size == 0:
        return []

    proc = preprocess_kill_feed(crop)
    results = reader.readtext(proc, detail=1, paragraph=False)
    detections = []

    for box, text, conf in results:
        text = text.strip()
        if conf < 0.35 or len(text) < 2:
            continue
        # box is 4 points, use top left y to group rows
        
        scale = 3.0
        y = int(box[0][1] / scale) + oy  # y-coordinate of the top-left corner
        x = int(box[0][0] / scale) + ox  # x-coordinate of the top-left corner

        detection = KillFeedDetection(text=text, conf=conf, x=x + ox, y=y + oy, frame_idx=frame_idx)

        detections.append(detection)  # Adjust coordinates to original frame
        # Draw the bounding box on the original frame for visualization
        
        x1 = int(box[0][0] / scale) + ox
        y1 = int(box[0][1] / scale) + oy
        x2 = int(box[2][0] / scale) + ox
        y2 = int(box[2][1] / scale) + oy
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2) 

    return detections

def group_rows(detections, y_threshold=10):
    """ Groups KillFeedDetection objects based on their y-coordinates. Returns a KillFeed object containing grouped KillFeedRow objects."""
    if not detections:
        empty_kill_feed = KillFeed()
        return empty_kill_feed

    # Sort top-to-bottom
    detections = sorted(detections, key=lambda r: r.y)

    groups = [[detections[0]]]

    for row in detections[1:]:
        # Compare to the average y of the current group
        avg_y = sum(r.y for r in groups[-1]) / len(groups[-1])

        if abs(row.y - avg_y) <= y_threshold:
            groups[-1].append(row)
        else:
            groups.append([row])

    kill_feed = KillFeed()

    for group in groups:
        # Sort left-to-right
        group.sort(key=lambda r: r.x)

        row = KillFeedRow(
            text=[r.text for r in group],
            y=int(sum(r.y for r in group) / len(group)),
            x=min(r.x for r in group),
            frame_idx= min(group[0], group[-1], key=lambda r: r.frame_idx).frame_idx,  # Use the frame index of the first detection in the group
            parts=group
        )
        kill_feed.add_row(row)

    return kill_feed

def normalize_text(text):
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

def is_user_name_match(detected_text, user_name):
    detected = normalize_text(detected_text)
    expected = normalize_text(user_name)

    if not detected or not expected:
        return False

    # Exact match
    if detected == expected:
        return True

    # Never allow the username to match as part of a longer word.
    # Prevents "aime" -> "me".
    if len(detected) > len(expected) + 1:
        return False

    # For short usernames, compare character-by-character.
    if len(expected) <= 3:
        # Allow one OCR character substitution
        if len(detected) == len(expected):
            differences = sum(
                a != b
                for a, b in zip(detected, expected)
            )

            return differences <= 1

        # Allow one extra OCR character
        if len(detected) == len(expected) + 1:
            differences = 0

            for i in range(len(detected)):
                shortened = detected[:i] + detected[i + 1:]

                if shortened == expected:
                    return True

            return False

        return False

    # Longer usernames can use fuzzy matching
    similarity = SequenceMatcher(
        None,
        detected,
        expected
    ).ratio()

    return similarity >= 0.70

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

    user_name = input("Enter your in-game name (or leave blank to skip): ").strip()
    kill_candidates = []
    user_kills = []

    print("Processing video... Press 'q' to stop.")

    paused = False  # Variable to track pause state
    frame_idx = 0  # Variable to track the current frame index
    step = 1  # Variable to control frame stepping
    rewind_step = 30  # Number of frames to rewind 
    ff_step = 30  # Number of frames to fast forward 
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not retrieve FPS from video.")
        return
    
    # Loop through the video frame by frame
    while True:

        if not paused:
            #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Set the current frame position
            ret, frame = cap.read()  # ret is a boolean that is True if the frame was read correctly

            # If ret is False, we've reached the end of the video
            if not ret:
                print("End of video reached.")
                break
            
            (crosshair_x, crosshair_y) = draw_crosshair(frame)

            # Run inference (detection) on the current frame
            enemy_results = enemy_model(frame, conf=0.5, verbose=False)
            
            # Process the results and draw boxes
            # The 'enemy_results' object contains all the bounding box coordinates for the enemy model
            for r in enemy_results:
                boxes = r.boxes
                
                closest_head = None
                closest_head_box = None
                closest_distance = float('inf')

                for box in boxes:
                    # Get the coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cropped = frame[y1:y2, x1:x2] # Crop the detected enemy region for head detection
                    if cropped.size == 0:
                        continue  # Skip if the cropped region is empty

                    draw_enemy(frame, box, enemy_class_names)

                    head_results = head_model(cropped, conf=0.3, imgsz= 320, verbose=False)
                    
                    head_found = False

                    best_head_box = None
                    best_head_conf = 0.0

                    for hr in head_results:
                        for head_box in hr.boxes:
                            conf = float(head_box.conf[0])
                            if conf > best_head_conf:
                                best_head_conf = conf
                                best_head_box = head_box
                            
                    #If head detected 
                    if best_head_box is not None:
                        head_found = True
                        (head_center_x, head_center_y), (hx1, hy1, hx2, hy2) = draw_head(frame, best_head_box, x1, y1, x2, y2)

                    if not head_found:
                        (head_center_x, head_center_y), (hx1, hy1, hx2, hy2) = draw_estimate_head(frame, (x1, y1, x2, y2))

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

            ocr_detections = ocr_kill_feed(frame, frame_idx)
            kill_feed = group_rows(ocr_detections)
            
            print(f"Frame {frame_idx}: Detected Kill Feed Rows: {kill_feed}")

            # Update kill_candidates list based on the current frame's kill feed
            for row in kill_feed.rows:

                matched = False
                
                for kill_candidate in kill_candidates:

                    y_difference = abs(row.y - kill_candidate.y)
                    if y_difference <= 20: # Same y
                        kill_candidate.last_frame = frame_idx
                        kill_candidate.rows.append(row)
                        matched = True
                        break
                    elif y_difference > 20: # Different y 
                        new_text = " ".join(row.text).lower().strip()
                        existing_text = " ".join(kill_candidate.rows[-1].text).lower().strip() #take last row of candidate text for now

                        text_match = False

                        if new_text == existing_text: # Same text (exact match)
                            text_match = True
                        else:
                            similarity = SequenceMatcher(None, new_text, existing_text).ratio()
                            """print(
                                    f"Comparing row={row.text} y={row.y} "
                                    f"against candidate={kill_candidate.rows[-1].text} "
                                    f"y={kill_candidate.y}"
                                )
                            print(f"Text similarity: {similarity:.3f}")"""

                            new_right = new_text.split()[-1] if new_text.split() else ""
                            existing_right = existing_text.split()[-1] if existing_text.split() else ""


                            right_similarity = SequenceMatcher(None, new_right, existing_right).ratio()

                            #print(f"Right word similarity: {right_similarity:.3f}")

                                
                            if (similarity >= 0.70) or (right_similarity >= 0.75): # Same text (fuzzy match)
                                text_match = True
                                #print(f"{Fore.GREEN}Fuzzy match found: {new_text} ~ {existing_text}")

                        if text_match == True: # Same text
                            if row.y < kill_candidate.y: # New row is above the candidate
                                print(f"{Fore.BLUE}Text match found AND row above kill candidate. KC ID: {kill_candidate.ID} row:{row.text}")
                                kill_candidate.rows.append(row)
                                kill_candidate.last_frame = frame_idx
                                kill_candidate.y = row.y  # Update the y-coordinate to the new row's y
                                
                                matched = True
                                break
                                
                            elif row.y > kill_candidate.y: # New row is below the candidate
                                matched = False
                                
                        else: # Different text
                            # If the new row is below the candidate, we can consider it a new kill candidate
                            if row.y > kill_candidate.y:
                                matched = False
                                
                if not matched:
                    new_kill_candidate = KillCandidate(
                                            rows=[row],
                                            x=row.x,
                                            y=row.y,
                                            first_frame=frame_idx,
                                            last_frame=frame_idx
                                        )
                    kill_candidates.append(new_kill_candidate)

            for kill_candidate in kill_candidates[:]:  # Iterate over a copy of the list to not modify it while iterating
                if (frame_idx - kill_candidate.first_frame) / fps > 5: # If the candidate is older than 5.6 seconds
                    print(f"{Fore.RED}Kill candidate expired: {kill_candidate}")
                                    
                    if not user_name:
                        continue

                    leftmost_part = min(kill_candidate.rows[-1].parts, key=lambda p: p.x)

                    if not is_user_name_match(leftmost_part.text, user_name): #Does user name match the leftmost part of the kill candidate row? If not, reject it.
                        print(
                            f"{Fore.RED}REJECTED USERNAME: "
                            f"detected={leftmost_part.text!r}, "
                            f"expected={user_name!r}"
                        )
                        kill_candidates.remove(kill_candidate)
                        continue
                    print(
                        f"{Fore.GREEN}ACCEPTED USERNAME: "
                        f"detected={leftmost_part.text!r}, "
                        f"expected={user_name!r}"
                        )
                    
                    user_kills.append(kill_candidate)
                    print(f"{Fore.YELLOW}NEW KILL: {kill_candidate}")

                    kill_candidates.remove(kill_candidate)

            print(f"Frame {frame_idx}: User Kills: {user_kills} \n          Kill Candidates: {kill_candidates}")

            # Display the frame on screen
            cv2.imshow('Valorant AI Coach - Vision Test', frame)
            frame_idx += step  # Move to the next frame

        key = cv2.waitKey(0 if paused else 1) & 0xFF

        if key == ord(' '):
            paused = not paused
        elif key == 2 or key == ord('a'):  # Left arrow key or 'a' key **arrow keys do not work on windows
            frame_idx = max(0, frame_idx - rewind_step)  # Rewind
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            #paused = True  # Pause after rewinding
        elif key == 3 or key == ord('d'):  # Right arrow key or 'd' key **arrow keys do not work on windows
            frame_idx += ff_step  # Fast forward
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            #paused = True  # Pause after fast forwarding
        elif key == ord('q'):
            print("Quitting video processing.")
            break  # Quit the loop
            

    # Clean up when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    MY_VIDEO = "input/test-clip-4.mp4"
    
    MY_ENEMY_MODEL = "runs/detect/valorant_coach/enemy_model_v1/weights/best.pt"
    MY_HEAD_MODEL = "runs/detect/valorant_coach/head_model_v1/weights/best.pt"
    
    process_valorant_replay(MY_VIDEO, MY_ENEMY_MODEL, MY_HEAD_MODEL)