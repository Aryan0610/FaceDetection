# Import necessary libraries
import cv2
import mediapipe as mp

# Initialize MediaPipe Object Detector
mp_object_detection = mp.solutions.face_detection
detector = mp_object_detection.FaceDetection(model_selection = 1)

# # Open the video file
# video = cv2.VideoCapture('videos/video1.mp4')

# Create a VideoCapture object to access the camera
video = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not video.isOpened():
    raise IOError("Cannot open webcam")

# Check if the video opened successfully
if not video.isOpened():
    print("Error opening video file")

# Read and process each frame of the video
while video.isOpened():
    ret, frame = video.read()

    # Check if there are no more frames to read
    if not ret:
        break

    # Convert BGR image to RGB
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform processing on the frame here
    results = detector.process(frame)

    # Draw bounding boxes around detected faces
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Create a window with a suitable name and size
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

    # Resize the window to a specific size
    # cv2.resizeWindow("Video", 450, 800)
    cv2.resizeWindow("Video", 800, 450)

    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video object and close all windows
video.release()
cv2.destroyAllWindows()