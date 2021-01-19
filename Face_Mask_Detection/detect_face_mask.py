import numpy as np
import cv2
import pyvirtualcam

IMG_W = 1280
IMG_H = 720

# Read video
# 0 == laptop cam, 1 == droid cam, 2 == virtual cam (don't use, since it's output)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier("data\\xml\\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("data\\xml\\haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("data\\xml\\haarcascade_mcs_mouth.xml")
upper_body = cv2.CascadeClassifier("data\\xml\\haarcascade_upperbody.xml")

# Adjust threshold value in range 80 to 105 based on your light.
bw_threshold = 80

# User message
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
weared_mask_font_color = (255, 255, 255)
not_weared_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
weared_mask = "Thank You for wearing a real MASK"
not_weared_mask = "Please wear MASK to defeat Corona"

cam = None


def virtualcam():
    with pyvirtualcam.Camera(width=1280, height=720, fps=30) as cam:
        print(cam)
        while True:
            frame = np.zeros((cam.height, cam.width, 4), np.uint8)  # RGBA
            frame[:, :, :3] = cam.frames_sent % 255  # grayscale animation
            frame[:, :, 3] = 255
            cam.send(frame)
            cam.sleep_until_next_frame()


def detect_face_mask():
    # uncomment to send to virtual cam:
    # cam = pyvirtualcam.Camera(width=IMG_W, height=IMG_H, fps=30)

    while 1:
        # Get individual frame
        ret, img = cap.read()
        img = cv2.flip(img, 1)

        # Convert Image into gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert image in black and white
        (thresh, black_and_white) = cv2.threshold(
            gray, bw_threshold, 255, cv2.THRESH_BINARY
        )
        # cv2.imshow('black_and_white', black_and_white)

        # detect face
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Face prediction for black and white
        faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)

        if len(faces) == 0 and len(faces_bw) == 0:
            cv2.putText(
                img,
                "No face found...",
                org,
                font,
                font_scale,
                weared_mask_font_color,
                thickness,
                cv2.LINE_AA,
            )
        elif len(faces) == 0 and len(faces_bw) == 1:
            # It has been observed that for white mask covering mouth, with gray image face prediction is not happening
            cv2.putText(
                img,
                weared_mask,
                org,
                font,
                font_scale,
                weared_mask_font_color,
                thickness,
                cv2.LINE_AA,
            )
        else:
            # Draw rectangle on gace
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                roi_gray = gray[y : y + h, x : x + w]
                roi_color = img[y : y + h, x : x + w]

                # Detect lips counters
                mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)

            # Face detected but Lips not detected which means person is wearing mask
            if len(mouth_rects) == 0:
                cv2.putText(
                    img,
                    weared_mask,
                    org,
                    font,
                    font_scale,
                    weared_mask_font_color,
                    thickness,
                    cv2.LINE_AA,
                )
            else:
                for mouth_rect in mouth_rects:
                    mx, my, mw, mh = mouth_rect

                    # Calling our function to add the mouthmask
                    img = add_mouth_mask(mouth_rect, img)

                    if y < my < y + h:
                        # Face and Lips are detected but lips coordinates are within face cordinates which `means lips prediction is true and
                        # person is not wearing a mask
                        cv2.putText(
                            img,
                            not_weared_mask,
                            org,
                            font,
                            font_scale,
                            not_weared_mask_font_color,
                            thickness,
                            cv2.LINE_AA,
                        )

                        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (0, 0, 255), 3)
                        break

        # Show frame with results
        cv2.imshow("Mask Detection", img)

        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

        # Send to virtual webcam
        if cam != None:
            cam.send(img_rgba)
            cam.sleep_until_next_frame()

        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            # Release video
            cap.release()
            cv2.destroyAllWindows()
            break


def add_mouth_mask(mouth_rect, img):
    mx, my, mw, mh = mouth_rect
    mask_x, mask_y, mask_w, mask_h = (
        int(mx - mw / 2),
        int(my - mh / 2),
        int(mw * 2),
        int(mh * 2),
    )
    # Load our overlay image: mustache.png
    img_mask = cv2.imread("mondmasker.png", -1)

    # Create the mask for the mustache
    orig_mask = img_mask[:, :, 3]

    # Create the inverted mask for the mustache
    orig_mask_inv = cv2.bitwise_not(orig_mask)

    # Convert mustache image to BGR
    # and save the original image size (used later when re-sizing the image)
    img_mask = img_mask[:, :, 0:3]
    rows, cols = img_mask.shape[:2]

    roi = img[
        mask_y : mask_y + mask_h, mask_x : mask_x + mask_w
    ]  # [[array(y)][array(x)]]

    dim = (mask_w, mask_h)
    resized_img_mask = cv2.resize(img_mask, dim, interpolation=cv2.INTER_AREA)
    # ---------------------------------------------
    img[mask_y : mask_y + mask_h, mask_x : mask_x + mask_w] = cv2.add(
        roi, resized_img_mask
    )

    img_with_mask = img

    return img_with_mask


detect_face_mask()