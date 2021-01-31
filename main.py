from functionality.functions import *

cap = cv2.VideoCapture(1)
faces = []
fourcc = cv2.VideoWriter_fourcc(*"XVID")
# out = cv2.VideoWriter("out.avi", fourcc, 20.0, (640, 480))


if __name__ == "__main__":
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Convert image into gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = caffe_detect_faces(frame, faces)
        faces = detect_mask_with_model(faces)
        frame = draw_on_frame(frame, faces)

        cv2.imshow("window", frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            # Release video
            cap.release()
            cv2.destroyAllWindows()
            break

    print("bye")
