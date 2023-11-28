import cv2
import dlib

# Initialize dlib's face detector and create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the image to be placed on the eyes
image = cv2.imread("image.jpg")

# Set the desired width and height for the image
new_width = 200
new_height = 100
# Resize the image to the desired size
image = cv2.resize(image, (new_width, new_height))

# Start the webcam
video_capture = cv2.VideoCapture(0)

# Set the initial position of the image
x = 0
y = 0

while True:
    # Get the current frame from the webcam
    _, frame = video_capture.read()

    # Use dlib's face detector to detect faces in the current frame
    faces = detector(frame)

    # Check if any face is detected
    if len(faces) > 0:
        # Get the first detected face
        face = faces[0]

        # Get the facial landmarks for the face
        landmarks = predictor(frame, face)

        # Get the eye landmarks
        left_eye = landmarks.part(36)
        right_eye = landmarks.part(45)

        # Calculate the position to place the image on the eyes
        x = left_eye.x
        y = left_eye.y

    # Calculate the coordinates for placing the image based on the eye position
    image_x = x - int(new_width / 2)
    image_y = y - int(new_height / 2)

    # Calculate the maximum coordinates to prevent the image from going off the frame
    max_x = frame.shape[1] - new_width
    max_y = frame.shape[0] - new_height

    # Adjust the image coordinates if they exceed the maximum values
    image_x = min(max_x, max(0, image_x))
    image_y = min(max_y, max(0, image_y))

    # Calculate the end coordinates for placing the image
    image_end_x = image_x + new_width
    image_end_y = image_y + new_height

    # Place the image on the frame
    frame[image_y:image_end_y, image_x:image_end_x] = image

    # Display the frame with the image on the eyes
    cv2.imshow("Eye Tracking with Image", frame) # frame

    # Check if the user pressed "q" to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()