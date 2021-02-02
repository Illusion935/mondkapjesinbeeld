import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import emoji
import os
import tensorflow as tf
from tensorflow import keras
import random
import tkinter as tk
from tkinter import simpledialog


from functionality.interface import *
from .Face import Face
from config.config import *

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
font = cv2.FONT_HERSHEY_DUPLEX
weared_mask_font_color = (255, 255, 255)
not_weared_mask_font_color = (0, 0, 255)


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
    ALLOWED_DIFF = 150
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
            if confidence > conf_bar:
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
                pass
            else:
                print(e)
        updated_faces.append(face)
    return updated_faces


def draw_on_frame(frame, faces, gebruiker_input):
    scalar = 200
    frame_h, frame_w = frame.shape[:2]
    top_message = gebruiker_input
    bottom_message = "Dit beeld wordt niet opgeslagen"
    for face in faces:
        x, y, w, h = face.roi

        # Top message
        frame = put_text(
            frame,
            top_message,
            (10, 25),
            color_RGB=(247, 226, 92),
            thickness=2,
            line=cv2.LINE_4,
        )

        # Bottom message
        btm_x, btm_y = calculate_bottom_text_pos((frame_w, frame_h), bottom_message)
        frame = put_text(
            frame,
            bottom_message,
            (btm_x, btm_y),
        )

        if face.done_calculating == True:
            if face.mask_detected == True:
                text = "Mondmasker gevonden!"
                frame = put_text(
                    frame,
                    text,
                    (x - int(w / 3), y),
                    scale=w / scalar,
                    color_RGB=(0, 255, 0),
                )
                frame = draw_smiley(frame, face.roi, face.positive_emoji_img)
            elif face.mask_detected == False:
                text = "Mondmasker vergeten!"
                frame = put_text(
                    frame,
                    text,
                    (x - int(w / 3), y),
                    scale=w / scalar,
                    color_RGB=(220, 5, 7),
                )
                frame = draw_smiley(frame, face.roi, face.negative_emoji_img)

    return frame


def calculate_bottom_text_pos(frame_dim, text):
    frame_w, frame_h = frame_dim
    x = int(frame_w - len(text) * 17)
    y = int(frame_h - frame_h / 40)
    return (x, y)


def draw_smiley(frame, roi, emoji_img):
    x, y, w, h = roi
    startX = x
    startY = y
    endX = x + w
    endY = y + w
    emoji_width = endX - startX
    emoji_height = endY - startY

    emoji_img = cv2.resize(emoji_img, (emoji_width, emoji_height))
    roi_img = frame[startY:endY, startX:endX]

    try:
        # Create a mask of emoji_img and create its inverse mask
        img2gray = cv2.cvtColor(emoji_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # Now black-out the area of emoji in ROI
        img1_bg = cv2.bitwise_and(roi_img, roi_img, mask=mask_inv)
        # Take only region of emoji from emoji_img.
        img2_fg = cv2.bitwise_and(emoji_img, emoji_img, mask=mask)
        # Put emoji in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)
        frame[startY:endY, startX:endX] = dst
    except cv2.error as e:
        if e.code == cv2.Error.StsUnmatchedSizes:
            pass
        elif e.code == cv2.Error.StsAssert:
            pass
        else:
            print(e)
    return frame


def put_text(
    frame,
    text,
    org,
    scale=1,
    color_RGB=(0, 0, 0),
    thickness=1,
    line=cv2.LINE_AA,
):
    color_BGR = color_RGB[::-1]
    cv2.putText(
        frame,
        text,
        org,
        font,
        scale,
        color_BGR,
        thickness,
        line,
    )
    return frame