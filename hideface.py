import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Apply emoji on the faces
    for (x, y, w, h) in faces:
        # Load the emoji image
        emoji = cv2.imread(r"f1.jpg", cv2.IMREAD_UNCHANGED) # add the path of the image by which the face will be hidden

        # Resize the emoji to the face size
        emoji = cv2.resize(emoji, (w, h))

        # Overlay the emoji on the face
        img[y:y+h, x:x+w] = emoji

    # Display the output
    cv2.imshow('Emoji Face', img)

    # Stop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
