import cv2

#Cascade Classifier Object from haar frontal face
f_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

# read image 
img = cv2.imread('data/me.png')

#using the haar cascade frontalface get the coordinates  (x y w h) of where the face has been found
found_face_coordinates = f_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5 )

for x,y,w,h in found_face_coordinates:
    #draw the rectangle on the cordinates returned on the image
    draw_rect_on_found_faces_img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3 )


cv2.imshow("Face Detection Result",draw_rect_on_found_faces_img)
cv2.waitKey()
cv2.destroyAllWindows()

