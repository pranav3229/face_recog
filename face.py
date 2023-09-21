import cv2
import numpy as np
import face_recognition

# Load the image in RGB format
imgelon_rgb = face_recognition.load_image_file('harshitsaraf.jpg')

# Find face locations
face_locations = face_recognition.face_locations(imgelon_rgb)

if len(face_locations) > 0:
    # Assuming there's at least one face, take the first one
    face = face_locations[0]
    copy = imgelon_rgb.copy()
    # Draw the rectangle
    cv2.rectangle(copy, (face[3], face[0]), (face[1], face[2]), (255, 0, 255), 2)

    # Resize the image for display
    resize_factor = 0.1  # Adjust this factor to control the resize amount
    copy_resized = cv2.resize(copy, None, fx=resize_factor, fy=resize_factor)

    # Display the resized image with the bounding box
    # cv2.imshow('Face with Bounding Box', copy_resized)
else:
    print("No faces detected in the image.")

# Resize the original image for display
resize_factor = 0.1  # Adjust this factor to control the resize amount
img_resized = cv2.resize(cv2.cvtColor(imgelon_rgb, cv2.COLOR_RGB2BGR), None, fx=resize_factor, fy=resize_factor)

# Display the original image
# cv2.imshow('Original Image', img_resized)
# `cv2.waitKey(0)
# cv2.destroyAllWindows()`


train_encode = face_recognition.face_encodings(imgelon_rgb)[0]


# lets test an image
test = face_recognition.load_image_file('atish.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
test_encode = face_recognition.face_encodings(test)[0]
print(face_recognition.compare_faces([train_encode],test_encode))