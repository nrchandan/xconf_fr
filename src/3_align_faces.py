import cv2
import dlib
import numpy as np

import openface


imgDim = 96

dlib_shape_predictor_path = '../models/dlib/shape_predictor_68_face_landmarks.dat'

# aligning face using openface which in turn uses dlib
fa_openface = openface.AlignDlib(dlib_shape_predictor_path)


def alignFace_openface(image, detected_faces):
    alignedFaces = []
    for (x1, y1, x2, y2) in detected_faces:
        dlib_rect = dlib.rectangle(x1, y1, x2, y2)
        alignedFaces.append(
            fa_openface.align(
                imgDim,
                image,
                dlib_rect,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

    return alignedFaces


def detect_face_dlib(image):
    face_detector = dlib.get_frontal_face_detector()
    detected_faces = face_detector(image, 0)

    #if faces were not found, try finding faces in upsampled the image
    if detected_faces is None:
        # Run the HOG face detector on the image data.
        # The result will be the bounding boxes of the faces in our image.
        detected_faces = face_detector(image, 1)

    return [(x.left(), x.top(), x.right(), x.bottom()) for x in detected_faces]


def webcam_FR():
    """
    Detect and recognize a face using a trained classifier.
    """

    # 0 for default video input (webcam)
    # Other options: webcam IP address, video files
    cap = cv2.VideoCapture(0)

    # VGA resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Capture frame-by-frame
        ret, image = cap.read()

        detected_faces = detect_face_dlib(image)

        aligned_faces = alignFace_openface(image, detected_faces)
        if aligned_faces:
            cv2.imshow('aligned_face', aligned_faces[0])

        # draw bounding boxes
        for (x1, y1, x2, y2) in detected_faces:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', image)

        # Wait for 1ms for input for keyboard input
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()


if __name__ == "__main__":
    webcam_FR()
    cv2.destroyAllWindows()

