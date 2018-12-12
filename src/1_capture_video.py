# OpenCV import
import cv2


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

