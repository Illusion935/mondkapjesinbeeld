import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import emoji
import os
import tensorflow as tf
from tensorflow import keras

from .Face import Face

face_cascade = cv2.CascadeClassifier("data\\xml\\haarcascade_frontalface_default.xml")
mouth_cascade = cv2.CascadeClassifier("data\\xml\\haarcascade_mcs_mouth.xml")

new_model = tf.keras.models.load_model("data/models/Aman_kumar_model.h5")
cvNet = cv2.dnn.readNetFromCaffe(
    "data/models/caffemodel/architecture.txt",
    "data/models/caffemodel/weights.caffemodel",
)

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


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


def caffe_detect_faces(frame, old_faces):
    gamma = 2.0
    updated_faces = []
    img_size = 124
    h = frame.shape[0]
    w = frame.shape[1]

    im = adjust_gamma(frame, gamma)
    im = cv2.resize(im, (300, 300))

    blob = cv2.dnn.blobFromImage(im, 1.0, (300, 300), (104.0, 177.0, 123.0))
    cvNet.setInput(blob)

    detections = cvNet.forward()
    for i in range(0, 10):
        try:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if endX - startX > 0.75 * w or endY - startY > 0.9 * h:
                break
            confidence = detections[0, 0, i, 2]
            if confidence > 0.11:
                roi = frame[startY:endY, startX:endX]
                roi = cv2.resize(roi, (img_size, img_size))
                roi = np.array(roi) / 255.0
                roi = roi.reshape(1, 124, 124, 3)
                result = new_model.predict(roi)
                print(result)

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        except:
            pass

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
        # updated_faces[-1] = detect_mouth(gray, updated_faces[-1])

    return updated_faces


def detect_mask(gray, faces):
    img_size = 124

    updated_faces = faces
    for i in range(0, len(faces)):
        x, y, w, h = faces[i].rio
        im = gray[y : y + h, x : x + w]
        im = cv2.resize(im, (img_size, img_size))
        im = np.array(im) / 255.0
        im = im.reshape(1, 124, 124, 3)
        result = new_model.predict(im)
        print(result)
        updated_faces[i] = detect_mouth(gray, faces[i])
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