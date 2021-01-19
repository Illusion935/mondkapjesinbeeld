from demo.demo import *

cap = cv2.VideoCapture(0)
faces = []

if __name__ == "__main__":
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Convert image into gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detect_faces(gray, faces)
        # print(faces)
        frame = draw_on_frame(frame, faces)

        cv2.imshow("window", frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            # Release video
            cap.release()
            cv2.destroyAllWindows()
            break

    print("bye")
    print("isa was here")
