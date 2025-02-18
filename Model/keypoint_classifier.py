#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='Model/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        input_details_tensor_index = self.input_details[0]['index']
        
        # Get expected input shape from the model
        input_shape = self.input_details[0]['shape']
        expected_features = input_shape[-1]  # Should be 84 in our case
        
        # If the model expects 84 features
        if expected_features == 84:
            if len(landmark_list) == 42:
                # If only one hand was detected, pad with zeros
                padded_landmarks = landmark_list + [0.0] * 42
                input_data = np.array(padded_landmarks, dtype=np.float32).reshape(1, 84)
            elif len(landmark_list) == 84:
                input_data = np.array(landmark_list, dtype=np.float32).reshape(1, 84)
            else:
                raise ValueError(f"Unexpected landmark_list length: {len(landmark_list)}")
        # If the model was trained with 42 features
        elif expected_features == 42:
            if len(landmark_list) == 84:
                input_data = np.array(landmark_list[:42], dtype=np.float32).reshape(1, 42)
            elif len(landmark_list) == 42:
                input_data = np.array(landmark_list, dtype=np.float32).reshape(1, 42)
            else:
                raise ValueError(f"Unexpected landmark_list length: {len(landmark_list)}")
        else:
            raise ValueError(f"Unsupported expected_features: {expected_features}")

        self.interpreter.set_tensor(input_details_tensor_index, input_data)
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))
        return result_index


def create_classifier_model():
    """Create model with input size for both hands"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(84,)),  # 42 landmarks * 2 hands
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(35, activation='softmax')  # A-Z + additional signs
    ])
    return model
