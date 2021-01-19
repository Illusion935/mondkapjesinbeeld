import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import emoji

from .Face import Face

face_cascade = cv2.CascadeClassifier("data\\xml\\haarcascade_frontalface_default.xml")
mouth_cascade = cv2.CascadeClassifier("data\\xml\\haarcascade_mcs_mouth.xml")

# Adjust threshold value in range 80 to 105 based on your light.
bw_threshold = 80

# User message
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
weared_mask_font_color = (255, 255, 255)
font_color = (255, 0, 0)
not_weared_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 0.5


def draw_on_frame(frame, faces):
    for face in faces:
        x, y, w, h = face.rio
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        if face.done_calculating == True:
            if face.mask_detected == True:
                frame = draw_smiley(frame, face.rio, ":smiley:")
            elif face.mask_detected == False:
                frame = draw_smiley(frame, face.rio, ":persevere:")
        else:
            cv2.putText(
                frame[int(y - h / 3) : y + h, x : x + 5 * w],
                "calculating",
                org,
                font,
                font_scale,
                font_color,
                thickness,
                cv2.LINE_AA,
            )

    return frame


def detect_faces(gray, old_faces, MIN_NEIGHBOURS=5):
    updated_faces = []
    ALLOWED_DIFF = 30

    faces_rio = face_cascade.detectMultiScale(gray, 1.1, MIN_NEIGHBOURS)
    if len(faces_rio) == 0:
        thresh, black_and_white = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        faces_rio = face_cascade.detectMultiScale(black_and_white, 1.1, 4)

    for rio in faces_rio:
        updated_faces.append(Face(rio))

        for face in old_faces:
            if (
                abs(face.rio[0] - rio[0]) < ALLOWED_DIFF
                and abs(face.rio[1] - rio[1]) < ALLOWED_DIFF
            ):
                updated_faces.pop(-1)
                face.rio = rio
                updated_faces.append(face)
                break
        updated_faces[-1] = detect_mouth(gray, updated_faces[-1])

    return updated_faces


def detect_mouth(gray, face, MIN_NEIGHBOURS=5):
    x, y, w, h = face.rio
    lower_face_roi_gray = gray[
        int(y + h / 2) : y + h,
        x : x + w,
    ]

    mouth_rects = mouth_cascade.detectMultiScale(
        lower_face_roi_gray, 1.5, MIN_NEIGHBOURS
    )
    if len(mouth_rects) > 0 and face.done_calculating == False:
        face.track_mask_detections("No mask")
    elif face.done_calculating == False:
        face.track_mask_detections("Mask")
    return face


def draw_smiley(frame, rio, emoticon):
    x, y, w, h = rio
    smiley = str(emoji.emojize(emoticon, use_aliases=True))
    FONT_SIZE = int(h / 2)
    y_above_face = int(y - FONT_SIZE * 0.86)
    if y_above_face > 0:
        face_rio_PIL = Image.fromarray(frame[y_above_face:y, x : x + w])

        font_PIL = ImageFont.truetype("OpenSansEmoji.ttf", FONT_SIZE)
        x_draw = int(w / 2 - FONT_SIZE / 2)
        y_draw = int(face_rio_PIL.height - FONT_SIZE * 1.33 * 0.85)

        draw = ImageDraw.Draw(face_rio_PIL)
        draw.text(
            (x_draw, y_draw),
            smiley,
            (255, 255, 255),
            font=font_PIL,
        )
        frame[y_above_face:y, x : x + w] = np.array(face_rio_PIL)
    else:
        face_rio_PIL = Image.fromarray(frame[y : y + h, x : x + w])
        font_PIL = ImageFont.truetype("OpenSansEmoji.ttf", FONT_SIZE)
        draw = ImageDraw.Draw(face_rio_PIL)
        draw.text(
            (int(w / 2 - FONT_SIZE / 2), 0),
            smiley,
            (255, 255, 255),
            font=font_PIL,
        )
        frame[y : y + h, x : x + w] = np.array(face_rio_PIL)
    return frame