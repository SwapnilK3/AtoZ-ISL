#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Standard library imports
import copy
import argparse
import itertools
import os
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
import queue

# Third-party imports
import cv2 as cv
import google.generativeai as genai
import mediapipe as mp
import numpy as np
import pygame.mixer as mixer
import pyttsx3
import tkinter as tk
from cryptography.fernet import Fernet
from gtts import gTTS
from PIL import Image, ImageTk
from tkinter import ttk, simpledialog

# Local imports
from utils import CvFpsCalc
from Model import KeyPointClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.3)

    args = parser.parse_args()

    return args

# Add a global flag to control landmark drawing
SHOW_LANDMARKS = False
CAPTURE_RESULTS = True  # Whether to capture recognized signs
CAPTURE_DELAY = 5.0  # Seconds between captures
last_capture_time = 0  # Track the last time we captured a sign
current_sentence = []  # Store the captured letters/words

# Add these global variables near your other globals
FIRST_DETECTION_DELAY = 10.0  # 10-second delay before first detection
first_detection_done = False  # Flag to track if first detection has occurred
app_start_time = 0  # Track when the app started

def main():
    # Check for Gemini API key
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # Check if key exists in file
    try:
        with open('gemini_api_key.txt', 'r') as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        pass
    
    # If still no key, ask user
    if not api_key:
        # Create a temporary root window for the dialog
        temp_root = tk.Tk()
        temp_root.withdraw()  # Hide the root window
        
        # Show dialog asking for API key
        api_key = simpledialog.askstring(
            "Gemini API Key", 
            "Enter your Gemini API key for better translations\n(Leave blank to use fallback):",
            parent=temp_root
        )
        
        # Save the key if provided
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            try:
                with open('gemini_api_key.txt', 'w') as f:
                    f.write(api_key)
                print("API key saved to gemini_api_key.txt")
            except Exception as e:
                print(f"Could not save API key: {e}")
        
        # Destroy the temporary root
        temp_root.destroy()
    
    # Check for googletrans installation
    # Add this line to access the global variables
    global SHOW_LANDMARKS, CAPTURE_RESULTS, CAPTURE_DELAY, last_capture_time, current_sentence, first_detection_done, app_start_time
    
    # Set the application start time
    app_start_time = time.time()
    
    # Reset the flag
    first_detection_done = False
    
    # Create the UI
    ui = SignLanguageUI()
    current_sentence = []  # Reset sentence
    
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model loading ####################################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    # Load labels ######################################################################
    secret_key = b'GA3GTwyIDooBW5gHX6a5BiEbAKDoiHUlFuDFM5x8ufc='  # Replace with your actual key
    keypoint_classifier_labels = load_encrypted_labels(secret_key, 'Model/keypoint_classifier_label.enc')
    
    # FPS measurement module ###########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history ###############################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # ##################################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Key processing (ESC: exit) ###################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        
        # Add these key handlers
        if key == 8:  # Backspace - delete last letter
            if current_sentence:
                current_sentence.pop()
                ui.update_sentence(current_sentence)
                
        if key == 32:  # Space - add space
            ui.add_space()
            
        if key == 13:  # Enter - save sentence
            ui.save_sentence()
            
        number, mode = select_mode(key, mode)

        # Camera capture #############################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display

        debug_image = copy.deepcopy(image)

        # Add ROI drawing
        debug_image = draw_roi(debug_image, fps, mode, number)

        # Detection #####################################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # ##############################################################################
        if results.multi_hand_landmarks is not None:
            landmark_lists = []
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Calculate landmarks for each hand
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                landmark_lists.append(pre_processed_landmark_list)
                
                # Draw visualizations
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                
                # Draw landmarks only if SHOW_LANDMARKS is True
                if SHOW_LANDMARKS:
                    debug_image = draw_landmarks(debug_image, landmark_list)
                
            # Process combined landmarks for classification
            combined_landmarks = combine_hand_landmarks(landmark_lists)
            hand_sign_id = keypoint_classifier(combined_landmarks)
            
            # Capture result logic - add this section
            current_time = time.time()
            
            # Check if this is the first detection and if enough time has passed
            if not first_detection_done:
                # Calculate time since app started
                time_since_start = current_time - app_start_time
                
                # Show countdown on screen
                remaining = max(0, int(FIRST_DETECTION_DELAY - time_since_start))
                cv.putText(debug_image, f"Starting in: {remaining}s", (20, 120), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)
                
                # If enough time has passed, allow detection
                if time_since_start >= FIRST_DETECTION_DELAY:
                    first_detection_done = True
                    last_capture_time = current_time  # Reset capture timer
            
            # Only process captures if first detection delay is done
            elif CAPTURE_RESULTS and current_time - last_capture_time >= CAPTURE_DELAY:
                if 0 <= hand_sign_id < len(keypoint_classifier_labels):
                    recognized_sign = keypoint_classifier_labels[hand_sign_id]
                    # Don't add if it's the same as the last one
                    if not current_sentence or current_sentence[-1] != recognized_sign:
                        # Update UI with new letter
                        ui.update_word(recognized_sign)
                        last_capture_time = current_time
                        
            # Convert to relative coordinates and normalize
            pre_processed_point_history_list = pre_process_point_history(
                debug_image, point_history)

            # Drawing
            debug_image = draw_info_text(
                debug_image,
                brect,
                handedness,
                keypoint_classifier_labels[hand_sign_id],
                ""
            )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Draw the current sentence
        debug_image = draw_sentence(debug_image, current_sentence)

        # Update the video frame in the UI
        ui.update_video_frame(debug_image)

        # Update the UI
        if not ui.update():
            break

    cap.release()
    cv.destroyAllWindows()
    try:
        ui.root.destroy()  # Close the Tkinter window
    except:
        pass


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = (key - 48) + 26  # This will map 0-9 to numbers 0-9
    elif 65 <= key <= 90:  # A ~ Z uppercase
        number = key - 65  # This will map A-Z to numbers 0-25
    elif 97 <= key <= 122:  # a ~ z lowercase
        number = key - 97  # This will map a-z to numbers 0-25

    if key == 110:  # n
        mode = 0
    if key == 106:  # j
        mode = 1
    # if key == 104:  # h
    #     mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Key points
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to 1D list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalize
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to 1D list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


# def logging_csv(number, mode, landmark_lists):
#     if mode == 0:
#         pass
#     if mode == 1:
#         if 0 <= number <= 25:  # Letters A-Z
#             csv_path = 'model/AtoZ.csv'
#             with open(csv_path, 'a', newline="") as f:
#                 writer = csv.writer(f)
                
#                 # Handle both single and dual hand data
#                 if len(landmark_lists) == 2:
#                     combined_landmarks = []
#                     combined_landmarks.extend(landmark_lists[0])
#                     combined_landmarks.extend(landmark_lists[1])
#                     writer.writerow([number, *combined_landmarks])  # number 0-25 for A-Z
#                 elif len(landmark_lists) == 1:
#                     combined_landmarks = []
#                     combined_landmarks.extend(landmark_lists[0])
#                     padding = [0.0] * 42
#                     combined_landmarks.extend(padding)
#                     writer.writerow([number, *combined_landmarks])
#     return


def draw_landmarks(image, landmark_point):
    # Connection lines
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # Wrist 1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # Wrist 2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # Thumb: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # Thumb: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # Thumb: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # Index finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # Index finger: second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # Index finger: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # Index finger: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # Middle finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # Middle finger: second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # Middle finger: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # Middle finger: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # Ring finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # Ring finger: second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # Ring finger: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # Ring finger: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # Little finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # Little finger: second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # Little finger: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # Little finger: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Bounding rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 25:  # Changed from 9 to 25
            letter = chr(number + 65)  # Convert number to letter (A=0, B=1, etc.)
            cv.putText(image, f"Letter: {letter}", (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    # Add instructions for sentence building
    cv.putText(image, "SPACE: Add space | BACKSPACE: Delete last | ENTER: Save sentence", 
              (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    
    return image


def combine_hand_landmarks(landmarks_list):
    """Combine landmarks from both hands into single feature vector"""
    if len(landmarks_list) == 2:
        # Calculate mean x position for each hand's landmarks
        mean_x_positions = []
        for landmarks in landmarks_list:
            # Take every other value (x coordinates) from the flattened list
            x_coords = landmarks[::2]
            mean_x = sum(x_coords) / len(x_coords)
            mean_x_positions.append(mean_x)
        
        # Sort hands based on mean x position (left to right)
        if mean_x_positions[0] > mean_x_positions[1]:
            landmarks_list[0], landmarks_list[1] = landmarks_list[1], landmarks_list[0]
            
        # Combine landmarks
        combined = []
        combined.extend(landmarks_list[0])
        combined.extend(landmarks_list[1])
        return combined
    else:
        # For single hand, pad with zeros for second hand
        combined = []
        combined.extend(landmarks_list[0])
        combined.extend([0.0] * len(landmarks_list[0]))  # Pad with zeros
        return combined


def draw_roi(image, fps, mode, number):
    # Draw ROI rectangle
    roi_size = 300
    h, w = image.shape[:2]
    x = w//2 - roi_size//2
    y = h//2 - roi_size//2
    
    # Draw ROI box
    cv.rectangle(image, (x, y), (x + roi_size, y + roi_size), (0, 255, 0), 2)
    
    # Add instructions
    if mode == 1:  # Training mode
        cv.putText(image, "Place hand in box", (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if 0 <= number <= 25:
            letter = chr(number + 65)
            cv.putText(image, f"Training '{letter}'", (x, y-30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image


def load_encrypted_labels(secret_key, encrypted_file_path):
    f = Fernet(secret_key)
    with open(encrypted_file_path, 'rb') as file:
        encrypted_data = file.read()
    decrypted = f.decrypt(encrypted_data).decode('utf-8')
    # Split into separate labels (assuming one per line)
    labels = [line.strip() for line in decrypted.splitlines() if line.strip()]
    return labels


def draw_sentence(image, sentence):
    """Display the current sentence at the bottom of the screen"""
    if not sentence:
        return image
        
    # Join all captured signs
    text = ' '.join(sentence)
    
    # Draw at the bottom with black background
    h, w = image.shape[:2]
    text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    
    # Create background box
    cv.rectangle(image, 
                (10, h - 50), 
                (max(10 + text_size[0] + 10, w//2), h - 10),
                (0, 0, 0), 
                -1)
    
    # Write text
    cv.putText(image, text, (15, h - 20),
              cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return image


class SignLanguageUI:
    def __init__(self):
        # Add near the beginning of __init__
        self.callback_queue = queue.Queue()
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Adjust speaking rate
        
        # Get available voices
        voices = self.engine.getProperty('voices')
        self.default_voice = self.engine.getProperty('voice')
        
        # Add properties
        self.last_spoken_word = ""
        self.word_buffer = []
        
        # Create main window with minimum size
        self.root = tk.Tk()
        self.root.title("Sign Language Translator")
        self.root.geometry("1200x700")
        self.root.minsize(800, 600)  # Set minimum window size
        self.root.configure(bg="#f0f0f0")
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", font=("Arial", 12))
        self.style.configure("TLabel", background="#f0f0f0", font=("Arial", 12))
        
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Use PanedWindow for better resizing control
        self.paned_window = ttk.PanedWindow(self.main_container, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Create left frame for video feed
        self.video_frame = ttk.Frame(self.paned_window, width=640, height=480)
        
        # Create right frame for controls and text
        self.controls_frame = ttk.Frame(self.paned_window, padding=10)
        
        # Create canvas for video feed
        self.video_canvas = tk.Canvas(self.video_frame, bg="black", width=640, height=480)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add language dropdown at the top of controls
        self.language_frame = ttk.Frame(self.controls_frame)
        self.language_frame.pack(fill=tk.X, pady=(0, 10), anchor=tk.N)
        
        ttk.Label(self.language_frame, text="Translate to:").pack(side=tk.LEFT)
        self.language_var = tk.StringVar(value="Hindi")
        languages = ["Hindi", "Marathi"]
        self.language_combo = ttk.Combobox(
            self.language_frame,
            textvariable=self.language_var,
            values=languages,
            state="readonly",
            width=15
        )
        self.language_combo.pack(side=tk.LEFT, padx=5)
        self.language_combo.bind("<<ComboboxSelected>>", lambda e: self.translate_sentence())
        
        # Current word label
        ttk.Label(self.controls_frame, text="Current Word:").pack(anchor=tk.W, pady=(5, 5))
        self.current_word_var = tk.StringVar()
        self.current_word_label = ttk.Label(
            self.controls_frame, 
            textvariable=self.current_word_var,
            font=("Arial", 16, "bold")
        )
        self.current_word_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Current sentence frame (input)
        ttk.Label(self.controls_frame, text="Current Sentence:").pack(anchor=tk.W)
        self.sentence_frame = ttk.Frame(self.controls_frame)
        self.sentence_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        
        # Text widget for the sentence
        self.sentence_text = tk.Text(
            self.sentence_frame, 
            wrap=tk.WORD, 
            height=3,
            font=("Arial", 14)
        )
        self.sentence_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for text widget
        scrollbar = ttk.Scrollbar(self.sentence_frame, command=self.sentence_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sentence_text.config(yscrollcommand=scrollbar.set)
        
        # Translation section (immediately below the input)
        ttk.Label(self.controls_frame, text="Translation:").pack(anchor=tk.W)
        self.translation_frame = ttk.Frame(self.controls_frame)
        self.translation_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Text widget for translation
        self.translation_text = tk.Text(
            self.translation_frame, 
            wrap=tk.WORD, 
            height=3,
            font=("Arial", 14)
        )
        self.translation_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for translation text
        trans_scrollbar = ttk.Scrollbar(self.translation_frame, command=self.translation_text.yview)
        trans_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.translation_text.config(yscrollcommand=trans_scrollbar.set)
        
        # Button frames with grid layout for better organization
        # First row - basic controls
        self.buttons_frame = ttk.Frame(self.controls_frame)
        self.buttons_frame.pack(fill=tk.X, pady=5)
        
        self.speak_button = ttk.Button(
            self.buttons_frame, 
            text="Speak Sentence",
            command=self.speak_sentence
        )
        self.speak_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        self.clear_button = ttk.Button(
            self.buttons_frame, 
            text="Clear",
            command=self.clear_sentence
        )
        self.clear_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        self.save_button = ttk.Button(
            self.buttons_frame, 
            text="Save",
            command=self.save_sentence
        )
        self.save_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # Second row - translation controls
        self.translation_buttons_frame = ttk.Frame(self.controls_frame)
        self.translation_buttons_frame.pack(fill=tk.X, pady=5)
        
        self.translate_button = ttk.Button(
            self.translation_buttons_frame,
            text="Translate",
            command=self.translate_sentence
        )
        self.translate_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        self.speak_translation_button = ttk.Button(
            self.translation_buttons_frame,
            text="Speak Translation",
            command=self.speak_translation  # Updated to new method
        )
        self.speak_translation_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # Third row - word controls
        self.word_buttons_frame = ttk.Frame(self.controls_frame)
        self.word_buttons_frame.pack(fill=tk.X, pady=5)
        
        # Add space button
        self.space_button = ttk.Button(
            self.word_buttons_frame, 
            text="Add Space",
            command=self.add_space
        )
        self.space_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # Auto-speak checkbox
        self.auto_speak_var = tk.BooleanVar(value=True)
        self.auto_speak_check = ttk.Checkbutton(
            self.word_buttons_frame,
            text="Auto-speak",
            variable=self.auto_speak_var,
            command=self.toggle_auto_speak
        )
        self.auto_speak_check.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # Add speak current word button
        self.speak_word_button = ttk.Button(
            self.word_buttons_frame, 
            text="Speak Word",
            command=self.speak_current_word
        )
        self.speak_word_button.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Store current sentence
        self.current_sentence = []
        self.auto_speak = True  # Default to speaking words automatically
        
        # Add keyboard shortcuts
        self.root.bind('<space>', lambda e: self.add_space())
        self.root.bind('<BackSpace>', lambda e: self.handle_backspace())
        self.root.bind('<Return>', lambda e: self.save_sentence())
        self.root.bind('<Control-t>', lambda e: self.translate_sentence())
        self.root.bind('<Control-s>', lambda e: self.speak_sentence())
        
        # Create Gemini translator
        try:
            # Try to get API key from environment variable or file
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                try:
                    with open('gemini_api_key.txt', 'r') as f:
                        api_key = f.read().strip()
                except FileNotFoundError:
                    api_key = None
                    
            if api_key:
                genai.configure(api_key=api_key)
                # Try different models in order of preference
                try:
                    self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                    print("Using gemini-1.5-flash model")
                except Exception:
                    try:
                        self.gemini_model = genai.GenerativeModel('gemini-1.0-flash')
                        print("Using gemini-1.0-flash model")
                    except Exception:
                        try:
                            self.gemini_model = genai.GenerativeModel('gemini-pro')
                            print("Using gemini-pro model")
                        except Exception as e:
                            print(f"Could not initialize any Gemini model: {e}")
                            self.translation_available = False
                            return
                
                self.translation_available = True
            else:
                self.translation_available = False
                self.status_var.set("No API key - translation disabled")
        except Exception as e:
            print(f"Error initializing Gemini: {e}")
            self.translation_available = False
        
        # Start the UI update loop as non-blocking
        self.root.update()
        
        # Add frames to paned window
        self.paned_window.add(self.video_frame)
        self.paned_window.add(self.controls_frame)

        # Set the initial frame sizes
        self.root.update_idletasks()
        width = self.paned_window.winfo_width()
        self.video_frame.configure(width=int(width * 0.6))
        self.controls_frame.configure(width=int(width * 0.4))
    
    def handle_backspace(self, event=None):
        """Handle backspace key press"""
        if self.current_sentence:
            self.current_sentence.pop()
            self.update_sentence(self.current_sentence)
            return "break"  # Prevent default handling

    def translate_sentence(self):
        """Translate the current sentence using Gemini API"""
        if not self.translation_available:
            self.status_var.set("Translation not available - no API key")
            return
            
        text = ''.join(self.current_sentence)
        
        if not text.strip():
            self.status_var.set("No text to translate")
            return
            
        target_language = self.language_var.get()
        self.status_var.set(f"Translating to {target_language}...")
        self.root.update_idletasks()
        
        # Clear previous translation
        self.translation_text.delete(1.0, tk.END)
        self.translation_text.insert(tk.END, "Translating...")
        self.root.update_idletasks()
        
        # Run translation in a separate thread
        threading.Thread(
            target=self._do_translation,
            args=(text, target_language),
            daemon=True
        ).start()
    
    def _do_translation(self, text, target_language):
        """Perform translation in background thread"""
        try:
            # Simple prompt for translation
            prompt = f"Translate this English text to {target_language} and give only translation without formating: {text}"
            
            # Get response from Gemini - use the correct model name
            response = self.gemini_model.generate_content(prompt)
            translation = response.text.strip()
            
            # Schedule UI update through the queue instead of after()
            self.callback_queue.put(lambda: self._update_translation_result(translation))
                
        except Exception as e:
            print(f"Translation error: {e}")
            
            # Schedule error message through the queue
            self.callback_queue.put(lambda: self._update_translation_error(str(e)))

    def _update_translation_result(self, translation):
        """Update the UI with successful translation (called in main thread)"""
        self.translation_text.delete(1.0, tk.END)
        self.translation_text.insert(tk.END, translation)
        self.status_var.set("Translation complete")
        
    def _update_translation_error(self, error_msg):
        """Update the UI with translation error (called in main thread)"""
        self.translation_text.delete(1.0, tk.END)
        self.translation_text.insert(tk.END, f"Translation failed: {error_msg}")
        self.status_var.set("Translation error")
    
    def toggle_auto_speak(self):
        """Toggle automatic speaking of recognized words"""
        self.auto_speak = self.auto_speak_var.get()
        if self.auto_speak:
            self.status_var.set("Auto-speak enabled")
        else:
            self.status_var.set("Auto-speak disabled")
    
    def add_space(self):
        """Add space to the sentence"""
        self.current_sentence.append(" ")
        self.word_buffer = []  # Reset word buffer when adding space
        self.update_sentence(self.current_sentence)
        
        # Speak the previous word if auto-speak is enabled
        if self.auto_speak and len(self.word_buffer) > 0:
            last_word = ''.join(self.word_buffer)
            if last_word.strip():
                self.status_var.set(f"Speaking: {last_word}")
                threading.Thread(
                    target=self._speak_text,
                    args=(last_word,),
                    daemon=True
                ).start()
        
        # Clear word buffer after speaking
        self.word_buffer = []
    
    def update_word(self, letter):
        """Update with a single letter and build words"""
        if not letter:
            return
            
        # Update the current letter display
        self.current_word_var.set(letter)
        self.root.update_idletasks()
        
        # Add letter to the word buffer
        self.word_buffer.append(letter)
        
        # Add to sentence
        self.current_sentence.append(letter)
        self.update_sentence(self.current_sentence)
    
    def speak_sentence(self):
        """Speak the current sentence as a whole"""
        text = ''.join(self.current_sentence)
        
        if text.strip():
            self.status_var.set("Speaking sentence...")
            self.root.update_idletasks()
            
            # Use a thread to prevent UI freezing
            threading.Thread(
                target=self._speak_text,
                args=(text,),
                daemon=True
            ).start()
        else:
            self.status_var.set("No text to speak")
    
    def _speak_text(self, text):
        """Helper method to speak text in a separate thread"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            
            # Use the callback queue instead of direct UI updates
            self.callback_queue.put(lambda: self.status_var.set("Ready"))
        except Exception as e:
            print(f"Speech error: {e}")
            self.callback_queue.put(lambda: self.status_var.set(f"Speech error: {e}"))
    
    def clear_sentence(self):
        """Clear the current sentence"""
        self.current_sentence = []
        self.word_buffer = []  # Also clear word buffer
        self.sentence_text.delete(1.0, tk.END)
        self.status_var.set("Sentence cleared")
    
    def save_sentence(self):
        """Save the current sentence to a file"""
        if not self.current_sentence:
            self.status_var.set("No sentence to save")
            return
        
        text = ''.join(self.current_sentence)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sentences_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Sentence: {text}\n")
        
        self.status_var.set(f"Saved to {filename}")
    
    def update_sentence(self, sentence):
        """Update the sentence display"""
        self.current_sentence = sentence
        
        # Format for display - join without adding spaces between letters
        formatted_text = ''.join(sentence)
        
        self.sentence_text.delete(1.0, tk.END)
        self.sentence_text.insert(tk.END, formatted_text)
        
        self.root.update_idletasks()
    
    def update(self):
        """Update the UI - call this regularly from main loop"""
        # Process any pending UI update requests from other threads
        try:
            while not self.callback_queue.empty():
                callback = self.callback_queue.get_nowait()
                callback()
        except queue.Empty:
            pass
            
        # Normal UI update
        try:
            self.root.update()
        except tk.TclError:
            # Window was closed
            return False
        return True
    
    def update_video_frame(self, frame):
        """Update the video frame with the current OpenCV image"""
        # Convert OpenCV BGR image to RGB format for tkinter
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Convert to PhotoImage format
        img = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.video_canvas.config(width=w, height=h)
        self.video_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        
        # Keep a reference to prevent garbage collection
        self.photo = photo

    def speak_current_word(self):
        """Speak the current word"""
        word = self.current_word_var.get()
        if word:
            self.status_var.set(f"Speaking: {word}")
            self.root.update_idletasks()
            
            # Use a thread to prevent UI freezing
            threading.Thread(
                target=self._speak_text,
                args=(word,),
                daemon=True
            ).start()
        else:
            self.status_var.set("No current word to speak")
    
    # Add this method to the class
    def speak_translation(self):
        """Speak the translated text in the target language"""
        translation = self.translation_text.get(1.0, tk.END).strip()
        target_language = self.language_var.get().lower()
        
        if not translation:
            self.status_var.set("No translation to speak")
            return
            
        # Map language names to gTTS language codes
        lang_codes = {
            "hindi": "hi",
            "marathi": "mr"  # Note: Marathi support may be limited
        }
        
        if target_language in lang_codes:
            self.status_var.set(f"Speaking {target_language} translation...")
            self.root.update_idletasks()
            
            # Use a thread to prevent UI freezing
            threading.Thread(
                target=self._speak_translation_gtts,
                args=(translation, lang_codes[target_language]),
                daemon=True
            ).start()
        else:
            # Fallback to regular TTS for English
            self.status_var.set("Speaking translation...")
            threading.Thread(
                target=self._speak_text,
                args=(translation,),
                daemon=True
            ).start()

    def _speak_translation_gtts(self, text, lang_code):
        """Speak text using Google TTS in specified language"""
        try:
            # Initialize mixer if not already done
            if not hasattr(self, 'mixer_initialized'):
                mixer.init()
                self.mixer_initialized = True
                
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_filename = temp_file.name
                
            # Generate the speech and save to the temp file
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(temp_filename)
            
            # Play the audio
            mixer.music.load(temp_filename)
            mixer.music.play()
            
            # Wait for playback to finish
            while mixer.music.get_busy():
                time.sleep(0.1)
                
            # Clean up
            mixer.music.unload()
            os.remove(temp_filename)
            
            # Update status
            self.callback_queue.put(lambda: self.status_var.set("Ready"))
            
        except Exception as e:
            print(f"Translation speech error: {e}")
            self.callback_queue.put(lambda: self.status_var.set(f"Speech error: {e}"))


class GeminiTranslator:
    def __init__(self, api_key=None):
        """Initialize Gemini translator with API key"""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        if not self.api_key:
            print("Warning: No Gemini API key provided.")
            self.available = False
            return
            
        try:
            # Configure the Gemini API
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.0-pro')
            self.available = True
            print("Gemini API configured successfully")
        except Exception as e:
            print(f"Error initializing Gemini API: {e}")
            self.available = False


if __name__ == '__main__':
    main()
