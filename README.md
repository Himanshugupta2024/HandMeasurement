# HandMeasurement
import cv2
import mediapipe as mp

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
PALM_WIDTH_CM = 8.5  # Average palm width in centimeters

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

# Function to convert landmark coordinates to pixel coordinates
def landmark_to_pixel(landmark, frame_shape):
    return (int(landmark.x * frame_shape[1]), int(landmark.y * frame_shape[0]))

# Function to get hand measurements
def get_hand_measurements(hand_landmarks, frame_shape):
    wrist_coords = landmark_to_pixel(hand_landmarks[mp_hands.HandLandmark.WRIST], frame_shape)
    index_base_coords = landmark_to_pixel(hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP], frame_shape)
    pinky_base_coords = landmark_to_pixel(hand_landmarks[mp_hands.HandLandmark.PINKY_MCP], frame_shape)
    
    # Calculate distances in pixels
    palm_width_pixels = calculate_distance(index_base_coords, pinky_base_coords)
    palm_length_pixels = calculate_distance(wrist_coords, index_base_coords)
    
    # Calculate pixel to cm ratio
    pixel_to_cm_ratio = PALM_WIDTH_CM / palm_width_pixels
    
    # Convert measurements to cm
    palm_width_cm = palm_width_pixels * pixel_to_cm_ratio
    palm_length_cm = palm_length_pixels * pixel_to_cm_ratio

    # Calculate actual finger lengths in cm using the orange points
    thumb_tip_coords = landmark_to_pixel(hand_landmarks[mp_hands.HandLandmark.THUMB_TIP], frame_shape)
    index_tip_coords = landmark_to_pixel(hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP], frame_shape)
    middle_tip_coords = landmark_to_pixel(hand_landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP], frame_shape)
    ring_tip_coords = landmark_to_pixel(hand_landmarks[mp_hands.HandLandmark.RING_FINGER_TIP], frame_shape)
    little_tip_coords = landmark_to_pixel(hand_landmarks[mp_hands.HandLandmark.PINKY_TIP], frame_shape)
    
    thumb_length_cm = calculate_distance(wrist_coords, thumb_tip_coords) * pixel_to_cm_ratio
    index_length_cm = calculate_distance(wrist_coords, index_tip_coords) * pixel_to_cm_ratio
    middle_length_cm = calculate_distance(wrist_coords, middle_tip_coords) * pixel_to_cm_ratio
    ring_length_cm = calculate_distance(wrist_coords, ring_tip_coords) * pixel_to_cm_ratio
    little_length_cm = calculate_distance(wrist_coords, little_tip_coords) * pixel_to_cm_ratio
    
    return {
        "Palm Width": palm_width_cm,
        "Palm Length": palm_length_cm,
        "Thumb Length": thumb_length_cm,
        "Index Finger Length": index_length_cm,
        "Middle Finger Length": middle_length_cm,
        "Ring Finger Length": ring_length_cm,
        "Little Finger Length": little_length_cm
    }

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get hand measurements
            measurements = get_hand_measurements(hand_landmarks.landmark, frame.shape)
            
            # Display measurements on the frame
            y_offset = 30
            for key, value in measurements.items():
                cv2.putText(frame, f'{key}: {value:.2f} cm', (10, y_offset), FONT, 1, (255, 0, 0), 2, cv2.LINE_AA)
                y_offset += 30

    # Display the frame
    cv2.imshow('Hand Measurement', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
