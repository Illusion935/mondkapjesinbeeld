import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import emoji
import os
import tensorflow as tf
from tensorflow import keras
import random

from .Face import Face

face_cascade = cv2.CascadeClassifier("data\\xml\\haarcascade_frontalface_default.xml")
mouth_cascade = cv2.CascadeClassifier("data\\xml\\haarcascade_mcs_mouth.xml")

aman_kumar_model = tf.keras.models.load_model("data/models/Aman_kumar_model.h5")
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


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


# load emoji's
positive_emojis = load_images_from_folder("data/emojis/positive")
negative_emojis = load_images_from_folder("data/emojis/negative")


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
            if confidence > 0.17:
                roi = [startX, startY, endX - startX, endY - startY]
                roi_img = frame[startY:endY, startX:endX]
                pos_emoji = random.choice(positive_emojis)
                neg_emoji = random.choice(negative_emojis)
                updated_faces.append(Face(roi, pos_emoji, neg_emoji, roi_img))
                for face in old_faces:
                    if (
                        abs(face.roi[0] - roi[0]) < ALLOWED_DIFF
                        and abs(face.roi[1] - roi[1]) < ALLOWED_DIFF
                    ):
                        updated_faces.pop(-1)
                        face.roi = roi
                        face.roi_img = roi_img
                        updated_faces.append(face)
                        old_faces.remove(face)
                        break

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
                old_faces.remove(face)
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
            roi_img = roi_img.reshape(1, img_size, img_size, 3)
            result = aman_kumar_model.predict(roi_img)
            result = result[0][0]
            if result < 0.11:
                face.mask_detected = True
            elif result >= 0.11:
                face.mask_detected = False
            face.done_calculating = True
        except cv2.error as e:
            if e.code == cv2.Error.StsAssert:
                print(roi_img.shape)
            else:
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
        face.count_mask_detections("No mask")
    elif face.done_calculating == False:
        face.count_mask_detections("Mask")
    return face


def draw_on_frame(frame, faces):
    for face in faces:
        x, y, w, h = face.roi
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        if face.done_calculating == True:
            if face.mask_detected == True:
                frame = draw_smiley(frame, face.roi, face.positive_emoji_img)
            elif face.mask_detected == False:
                frame = draw_smiley(frame, face.roi, face.negative_emoji_img)
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


def draw_smiley(frame, roi, emoji_img):
    x, y, w, h = roi
    startX = x
    endX = x + w
    startY = y
    endY = y + w
    emoji_width = endX - startX
    emoji_height = endY - startY

    emoji_img = cv2.resize(emoji_img, (emoji_width, emoji_height))
    roi_img = frame[startY:endY, startX:endX]
    # Create a mask of emoji_img and create its inverse mask
    img2gray = cv2.cvtColor(emoji_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    try:
        # Now black-out the area of emoji in ROI
        img1_bg = cv2.bitwise_and(roi_img, roi_img, mask=mask_inv)
        # Take only region of emoji from emoji_img.
        img2_fg = cv2.bitwise_and(emoji_img, emoji_img, mask=mask)
        # Put emoji in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)
        frame[startY:endY, startX:endX] = dst
    except cv2.error as e:
        if e.code == cv2.Error.StsUnmatchedSizes:
            print("frame: ", img1_bg.shape)
            print("emoji img: ", img2_fg.shape)
            pass
        elif e.code == cv2.Error.StsAssert:
            print("roi: ", roi_img.shape)
        else:
            print(e)
    return frame