import cv2

f_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

while True:
    # capture
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found_faces_cordinates = f_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # Draw a rectangle around the found faces cordinateses
    for (x, y, w, h) in found_faces_cordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Real-time face detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()