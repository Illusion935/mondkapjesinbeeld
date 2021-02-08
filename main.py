from functionality.functions import *
from functionality.interface import *
from config.config import *

dim = (1080, 720)
cap = cv2.VideoCapture(cam_id)
faces = []
fourcc = cv2.VideoWriter_fourcc(*"XVID")
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
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        if webcam_check_failed(frame):
            show_error("Webcam not found")
            break
        display_fps(frame)
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
