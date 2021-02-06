from functionality.functions import *
from functionality.interface import *
from config.config import *

cap = cv2.VideoCapture(cam_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
faces = []
fourcc = cv2.VideoWriter_fourcc(*"XVID")
dim = (1080, 720)
# out = cv2.VideoWriter("out.avi", fourcc, 20.0, dim)


def show_error(text):
    print(text)


def webcam_check_failed(frame):
    if frame is None:
        return True
    else:
        return False


if __name__ == "__main__":
    # Check if webcam can be found
    frame = cap.read()[1]
    if webcam_check_failed(frame):
        show_error("Webcam not found")
        assert frame, "Webcam not found"

    cv2.namedWindow("Mondkapjes in Beeld", cv2.WINDOW_FREERATIO)

    # Ask for user input for the message of the day
    gebruiker_input = interface()
    while True:
        ret, frame = cap.read()
        if webcam_check_failed(frame):
            show_error("Webcam not found")
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        # Convert image into gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = caffe_detect_faces(frame, faces)
        faces = detect_mask_with_model(faces)
        frame = draw_on_frame(frame, faces, gebruiker_input)
        cv2.imshow("Mondkapjes in Beeld", frame)
        k = cv2.waitKey(30) & 0xFF

        # Exit program with ESC key
        if k == 27:
            break
    # Release video
    cap.release()
    cv2.destroyAllWindows()
    print("bye")
