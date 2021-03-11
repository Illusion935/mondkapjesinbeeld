from functionality.functions import *
from config.config import *

dim = (1080, 720)
cap = cv2.VideoCapture(cam_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
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


def LAB_CLAHE_contrast_improvement(img):
    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


if __name__ == "__main__":
    # Check if webcam can be found
    frame = cap.read()[1]
    if webcam_check_failed(frame):
        show_error("Webcam not found")
        assert frame, "Webcam not found"

    cv2.namedWindow("Mondkapjes in Beeld", cv2.WINDOW_FREERATIO)

    while True:
        ret, frame = cap.read()
        if webcam_check_failed(frame):
            show_error("Webcam not found")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        dst = LAB_CLAHE_contrast_improvement(frame)

        faces = caffe_detect_faces(dst, faces)
        detect_mask_with_model(faces)
        draw_on_frame(frame, faces)
        cv2.imshow("Mondkapjes in Beeld", frame)
        k = cv2.waitKey(30) & 0xFF

        # Exit program with ESC key
        if k == 27:
            break
    # Release video
    cap.release()
    cv2.destroyAllWindows()
    print("bye")
