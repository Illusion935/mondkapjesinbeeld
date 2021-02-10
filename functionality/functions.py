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
import threading
from playsound import playsound
import time
from datetime import datetime


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


def play_sound(audiofile):
    playsound(audiofile)


# For FPS
start_time = None
frame_count = 0
total_seconds = 0
do_reset = True
fps = 0

# For playing audio's with interval
prev_time = time.time()


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
    return images


def load_audio_from_folder(folder):
    audio = os.listdir(folder)
    return audio


# load emoji's
positive_emojis = load_images_from_folder("data/emojis/positive")
negative_emojis = load_images_from_folder("data/emojis/negative")
morning_audio_files = load_audio_from_folder("data/audio/morgen")
afternoon_audio_files = load_audio_from_folder("data/audio/middag")
evening_audio_files = load_audio_from_folder("data/audio/avond")


def calculate_FPS():
    global start_time
    global frame_count
    global total_seconds
    global do_reset
    global fps
    if start_time is not None:
        end_time = time.time()
        seconds = end_time - start_time
        # Take average of 20 frames
        if frame_count >= 50:
            do_reset = True
        # Start counting after number of frames
        if frame_count >= 4 and do_reset == True:
            frame_count = 0
            total_seconds = 0
            do_reset = False
        frame_count += 1
        total_seconds += seconds
        fps = frame_count / total_seconds
    start_time = time.time()
    return (frame_count, fps)


def display_fps(frame):
    frame_count, fps = calculate_FPS()
    put_text(
        frame,
        "Calc. {0} frames".format(frame_count),
        (0, frame.shape[0] - 40),
    )
    put_text(frame, "{0} fps".format(round(fps, 2)), (0, frame.shape[0] - 10))


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))


def get_daytime():
    hour = datetime.now().hour
    return (
        "morning" if 5 <= hour < 12 else "afternoon" if 12 <= hour < 17 else "evening"
    )


def get_audio():
    daytime = get_daytime()
    if daytime == "morning":
        return "morgen/" + random.choice(morning_audio_files)
    elif daytime == "afternoon":
        return "middag/" + random.choice(afternoon_audio_files)
    elif daytime == "evening":
        return "avond/" + random.choice(evening_audio_files)


def caffe_detect_faces(frame, old_faces):
    global prev_time
    # Create thread for playing sound
    sound_thread = threading.Thread(target=play_sound, daemon=True)
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
                        face.new_face = False
                        face.roi = roi
                        face.roi_img = roi_img
                        # face.wait_till_delete = WAIT_FRAMES
                        updated_faces.append(face)
                        old_faces.remove(face)
                        break

                if updated_faces[-1].new_face == True:
                    current_time = time.time()
                    if not sound_thread.is_alive() and current_time > prev_time + 5:
                        audio = get_audio()
                        sound_thread = threading.Thread(
                            target=play_sound,
                            args=("data/audio/" + audio,),
                            daemon=True,
                        )
                        sound_thread.start()
                        prev_time = current_time
        except:
            pass

    # for face in old_faces:
    #     if face.wait_till_delete > 0:
    #         face.wait_till_delete -= 1
    #         updated_faces.append(face)
    #     else:
    #         old_faces.remove(face)

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


banner_x_offset = 0


def draw_on_frame(frame, faces, gebruiker_input):
    scalar = 170
    frame_h, frame_w = frame.shape[:2]
    top_message = gebruiker_input
    bottom_message = "Dit beeld wordt niet opgeslagen"

    for face in faces:
        x, y, w, h = face.roi

        if face.done_calculating == True:
            if face.mask_detected == True:
                frame = draw_smiley(frame, face.roi, face.positive_emoji_img)
                text = "Mondmasker gevonden!"
                frame = put_text(
                    frame,
                    text,
                    (x - int(w / 2), y),
                    scale=w / scalar,
                    # font_size=(int(w / 2)),
                    color_RGB=(0, 255, 0),
                )
            elif face.mask_detected == False:
                frame = draw_smiley(frame, face.roi, face.negative_emoji_img)
                text = "Mondmasker vergeten!"
                frame = put_text(
                    frame,
                    text,
                    (x - int(w / 2), y),
                    scale=w / scalar,
                    # font_size=(int(w / 2)),
                    color_RGB=(220, 5, 7),
                )

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

    return frame


def calculate_bottom_text_pos(frame_dim, text):
    frame_w, frame_h = frame_dim
    x = int(frame_w - len(text) * 17)
    y = int(frame_h - frame_h / 40)
    return (x, y)


def draw_smiley(frame, roi, emoji_BGRA):
    x, y, w, h = roi
    startX = x
    startY = y
    endX = x + w
    endY = y + w
    emoji_width = endX - startX
    emoji_height = endY - startY

    emoji_BGRA = cv2.resize(emoji_BGRA, (emoji_width, emoji_height))
    roi_img = frame[startY:endY, startX:endX]

    try:
        # Get the BGR image and the Alpha channel separate from the BGRA emoji image
        emoji_BGR = emoji_BGRA[:, :, :3]
        alpha = emoji_BGRA[:, :, 3]
        # Threshold based on the alpha channel of the emoji img.
        # It's thresholded at 200, because emojis sometimes have
        # some shadow with an Alpha higher than 0
        ret, mask = cv2.threshold(alpha, 150, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # # Now black-out the area of emoji in ROI
        img1_bg = cv2.bitwise_and(roi_img, roi_img, mask=mask_inv)
        # # Take only region of emoji from emoji_img.
        img2_fg = cv2.bitwise_and(emoji_BGR, emoji_BGR, mask=mask)
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
    color_RGB=(255, 255, 255),
    thickness=1,
    line=cv2.LINE_AA,
):
    font = cv2.FONT_HERSHEY_DUPLEX
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


# def put_text(
#     frame,
#     text,
#     org,
#     font_size=15,
#     color_RGB=(255, 255, 255),
#     thickness=1,
#     line=cv2.LINE_AA,
# ):
#     font = ImageFont.truetype("data/fonts/verdana.ttf", 15, 0)

#     frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#     # frame_PIL = Image.fromarray(frame_RGB)
#     # draw = ImageDraw.Draw(frame_PIL)
#     # draw.text(org, text, font=font, fill=color_RGB)

#     # frame_RGB = np.array(frame_PIL)
#     frame = cv2.cvtColor(frame_RGB, cv2.COLOR_RGB2BGR)

# return frame
