import numpy as np
import itertools
import copy

def calc_landmark_list(image, landmarks):
    """Convert landmarks from MediaPipe format to xyz coordinates"""
    image_width, image_height = image.shape[1], image.shape[0]
    
    landmark_point = []
    
    for _, landmark in enumerate(landmarks.landmark):
        # Extract x, y coordinates (normalized to image dimensions)
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        
        # Add to list
        landmark_point.append([landmark_x, landmark_y])
    
    return landmark_point



def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 相対座標に変換
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # 1次元リストに変換
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # 正規化
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list




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
