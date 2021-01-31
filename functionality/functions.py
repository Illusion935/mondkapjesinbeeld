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


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


def caffe_detect_faces(frame, old_faces):
    gamma = 2.0
    ALLOWED_DIFF = 30
    updated_faces = []
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
                roi = [startX, startY, endX - startX, endY - startY]
                roi_img = frame[startY:endY, startX:endX]
                updated_faces.append(Face(roi, roi_img))
                for face in old_faces:
                    if (
                        abs(face.roi[0] - roi[0]) < ALLOWED_DIFF
                        and abs(face.roi[1] - roi[1]) < ALLOWED_DIFF
                    ):
                        updated_faces.pop(-1)
                        face.roi = roi
                        face.roi_img = roi_img
                        updated_faces.append(face)
                        break

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        except:
            pass

    return updated_faces


def detect_faces(gray, old_faces, MIN_NEIGHBOURS=5):
    updated_faces = []
    ALLOWED_DIFF = 30

    faces_roi = face_cascade.detectMultiScale(gray, 1.1, MIN_NEIGHBOURS)
    if len(faces_roi) == 0:
        thresh, black_and_white = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        faces_roi = face_cascade.detectMultiScale(black_and_white, 1.1, 4)

    for roi in faces_roi:
        updated_faces.append(Face(roi))

        for face in old_faces:
            if (
                abs(face.roi[0] - roi[0]) < ALLOWED_DIFF
                and abs(face.roi[1] - roi[1]) < ALLOWED_DIFF
            ):
                updated_faces.pop(-1)
                face.roi = roi
                updated_faces.append(face)
                break

    return updated_faces


def detect_mask_with_model(old_faces):
    updated_faces = []
    img_size = 124

    for face in old_faces:
        try:
            roi_img = face.roi_img
            roi_img = cv2.resize(roi_img, (img_size, img_size))
            roi_img = np.array(roi_img) / 255.0
            roi_img = roi_img.reshape(1, 124, 124, 3)
            result = new_model.predict(roi_img)
            print(result)
            result = result[0][0]
            if result < 0.5:
                face.mask_detected = True
            elif result >= 0.5:
                face.mask_detected = False
            face.done_calculating = True
        except:
            pass
        updated_faces.append(face)
    return updated_faces


def detect_mask(gray, faces):
    img_size = 124

    updated_faces = faces
    for i in range(0, len(faces)):
        updated_faces[i] = detect_mouth(gray, faces[i])
    return updated_faces


def detect_mouth(gray, face, MIN_NEIGHBOURS=5):
    x, y, w, h = face.roi
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


def draw_on_frame(frame, faces):
    for face in faces:
        x, y, w, h = face.roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        if face.done_calculating == True:
            if face.mask_detected == True:
                frame = draw_smiley(frame, face.roi, ":smiley:")
            elif face.mask_detected == False:
                frame = draw_smiley(frame, face.roi, ":persevere:")
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


def draw_smiley(frame, roi, emoticon):
    x, y, w, h = roi
    smiley = str(emoji.emojize(emoticon, use_aliases=True))
    sizes = [20, 32, 40, 48, 64, 96, 160]
    FONT_SIZE = 109  # min(sizes, key=lambda x: abs(x - w))
    y_above_face = int(y - FONT_SIZE * 0.86)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if False:  # y_above_face > 0:
        face_roi_PIL = Image.fromarray(frame[y_above_face:y, x : x + w])

        font_PIL = ImageFont.truetype("SamsungColorEmoji.ttf", FONT_SIZE)
        x_draw = int(w / 2 - FONT_SIZE / 2)
        y_draw = int(face_roi_PIL.height - FONT_SIZE * 1.33 * 0.85)

        draw = ImageDraw.Draw(face_roi_PIL)
        draw.text(
            (x_draw, y_draw),
            smiley,
            (255, 255, 255),
            font=font_PIL,
        )
        frame[y_above_face:y, x : x + w] = np.array(face_roi_PIL)
    else:
        # try:
        face_roi_PIL = Image.fromarray(frame_rgb[y : y + h, x - w : x + 2 * w])
        font_PIL = ImageFont.truetype(
            "NotoColorEmoji.ttf",
            size=FONT_SIZE,
            layout_engine=ImageFont.LAYOUT_RAQM,
        )
        draw = ImageDraw.Draw(face_roi_PIL)
        draw.text(
            (int(w + w / 2 - FONT_SIZE / 2), 0),
            smiley,
            fill="#faa",
            embedded_color=True,
            font=font_PIL,
        )
        face_roi_BGR = cv2.cvtColor(np.array(face_roi_PIL), cv2.COLOR_RGB2BGR)
        frame[y : y + h, x - w : x + 2 * w] = face_roi_BGR
    # except:
    #     pass
    return frame