# import cv2
# import mediapipe as mp
# import pyautogui
#
# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#
# # Global variable to store the last X position
# last_position = None
#
#
# def check_movement(hand_landmarks):
#     """
#     Check the movement of the hand based on the X-coordinate of the hand landmarks.
#     """
#     global last_position
#     # Calculate the mean X-coordinate of all hand landmarks
#     mean_x = round(sum([landmark.x for landmark in hand_landmarks.landmark]) / len(hand_landmarks.landmark), 2)
#
#     # Compare with the last recorded position
#     if last_position is not None:
#         if mean_x < last_position:
#             print("Moving left")
#         elif mean_x > last_position:
#             print("Moving right")
#
#     # Update the last position
#     last_position = mean_x
#
#
# # Initialize Video Capture
# cap = cv2.VideoCapture(0)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Unable to read from camera.")
#         break
#
#     # Convert the frame to RGB and flip horizontally
#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     image_rgb = cv2.flip(image_rgb, 1)  # Flip for a mirrored view
#     frame = cv2.flip(frame, 1)
#
#     # Process the frame with MediaPipe Hands
#     results = hands.process(image_rgb)
#
#     # Check for detected hands
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Draw hand landmarks on the frame
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#             # Check hand movement
#             check_movement(hand_landmarks)
#
#     # Display the frame
#     cv2.imshow('Hand Movement Detection', frame)
#
#     # Exit the loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start Webcam Capture
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access the camera.")
        break

    # Convert the frame to RGB (required by MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    # Process the frame
    results = pose.process(frame_rgb)

    # Check if landmarks are detected
    if results.pose_landmarks:
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmarks
        landmarks = results.pose_landmarks.landmark

        # Get LEFT_SHOULDER and RIGHT_SHOULDER coordinates
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Convert normalized coordinates to pixel coordinates
        h, w, _ = frame.shape
        left_shoulder_pixel = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        right_shoulder_pixel = (int(right_shoulder.x * w), int(right_shoulder.y * h))

        # Draw circles on the shoulders
        cv2.circle(frame, left_shoulder_pixel, 10, (0, 255, 0),-1)
        cv2.circle(frame, right_shoulder_pixel, 10, (255, 0, 0), -1)

        # Print the coordinates of the shoulders
        print(f"Left Shoulder: {left_shoulder_pixel}, Right Shoulder: {right_shoulder_pixel}")

    # Display the frame
    cv2.imshow("MediaPipe Pose - Shoulder Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()
