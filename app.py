from flask import Flask, render_template, request, jsonify, url_for
from flask_socketio import SocketIO, emit
import os
import numpy as np
import mediapipe as mp
import cv2 as cv
import base64
from PIL import Image
import io
import time
import json
import tensorflow as tf
from collections import deque

# Import utility functions
from utils.landmark_utils import calc_landmark_list, pre_process_landmark, combine_hand_landmarks

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sign_language_translator_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize MediaPipe Hands - using your original settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3,
)

# Use your existing KeyPointClassifier implementation
class KeyPointClassifier:
    def __init__(self, model_path='model/keypoint_classifier/keypoint_classifier.tflite'):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Print model details for debugging
        print(f"Model loaded: {model_path}")
        print(f"Expected input shape: {self.input_details[0]['shape']}")
        
    def __call__(self, landmark_list):
        try:
            input_details_tensor_index = self.input_details[0]['index']
            
            # Debug input dimensions
            expected_shape = self.input_details[0]['shape']
            expected_dim = expected_shape[1] if len(expected_shape) > 1 else 84
            
            print(f"Model expects input shape: {expected_shape}, received: {len(landmark_list)}")
            
            # Ensure exactly 84 features as expected by the model
            if len(landmark_list) != expected_dim:
                print(f"WARNING: Resizing input from {len(landmark_list)} to {expected_dim}")
                if len(landmark_list) < expected_dim:
                    # Pad with zeros
                    landmark_list = landmark_list + [0.0] * (expected_dim - len(landmark_list))
                else:
                    # Truncate
                    landmark_list = landmark_list[:expected_dim]
            
            # For debugging only - print first few values of the input
            print(f"Input sample: {landmark_list[:10]}...")
            
            # Convert to numpy array and set tensor
            input_data = np.array([landmark_list], dtype=np.float32)
            self.interpreter.set_tensor(input_details_tensor_index, input_data)
            self.interpreter.invoke()
            
            # Get output
            output_details_tensor_index = self.output_details[0]['index']
            result = self.interpreter.get_tensor(output_details_tensor_index)
            
            # For debugging - print results for top 3 predictions
            top_indices = np.argsort(result[0])[-3:][::-1]
            for idx in top_indices:
                print(f"Prediction {idx} ({labels[idx] if idx < len(labels) else 'Unknown'}): {result[0][idx]:.4f}")
            
            result_index = np.argmax(result)
            confidence = result[0][result_index]
            
            return result_index, confidence
        except Exception as e:
            print(f"Error in model inference: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a fallback value when prediction fails
            return 0, 0.1  # Return first class with low confidence

# Load your specific encrypted labels
def load_labels():
    try:
        # For simplicity, using hardcoded labels for now
        return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    except Exception as e:
        print(f"Error loading labels: {e}")
        return ['Unknown']

# Initialize model and labels
keypoint_classifier = KeyPointClassifier()
labels = load_labels()

# Buffer for stability
recent_predictions = deque(maxlen=5)
MIN_DETECTION_CONFIDENCE = 0.50
PREDICTION_STABILITY_THRESHOLD = 3

def process_image_for_prediction(image_data):
    """Process image data and extract hand landmarks"""
    try:
        # Convert image to OpenCV format
        image = cv.cvtColor(np.array(image_data), cv.COLOR_RGB2BGR)
        
        # Process with MediaPipe
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        
        if not results.multi_hand_landmarks:
            print("No hands detected in the image")
            return None, None
        
        # Process landmarks from all detected hands
        landmark_lists = []
        handedness_info = []
        
        # Extract handedness information
        if results.multi_handedness:
            for hand_info in results.multi_handedness:
                label = hand_info.classification[0].label
                score = float(hand_info.classification[0].score)
                handedness_info.append({
                    'label': label,
                    'score': score
                })
                print(f"Detected {label} hand with confidence {score:.2f}")
        
        # Process hand landmarks using your original functions
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            print(f"Processing landmarks for hand {i+1}")
            # Calculate landmarks for each hand
            landmark_list = calc_landmark_list(image_rgb, hand_landmarks)
            print(f"Raw landmarks count: {len(landmark_list)}")
            
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            print(f"Pre-processed landmarks length: {len(pre_processed_landmark_list)}")
            
            landmark_lists.append(pre_processed_landmark_list)
        
        # Combine landmarks if multiple hands are detected
        combined_landmarks = combine_hand_landmarks(landmark_lists)
        if combined_landmarks is None:
            print("Failed to combine landmarks")
            return None, None
            
        print(f"Final combined landmarks length: {len(combined_landmarks)}")
        return combined_landmarks, handedness_info
        
    except Exception as e:
        print(f"Error in landmark processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def predict_sign(landmark_list, previous_detection=None, is_streaming=False):
    """Predict the sign based on landmarks"""
    if landmark_list is None:
        return "No Sign", -1, 0.0
    
    # Get prediction from the model
    hand_sign_id, confidence = keypoint_classifier(landmark_list)
    
    # Get the sign text from labels
    if 0 <= hand_sign_id < len(labels):
        current_sign_text = labels[hand_sign_id]
    else:
        current_sign_text = "Unknown"
    
    # Print detailed prediction info
    print(f"Raw prediction: {current_sign_text} (ID: {hand_sign_id}) with confidence {confidence:.2f}")
    
    # For single image mode, return the prediction directly
    if not is_streaming:
        return current_sign_text, hand_sign_id, float(confidence)
    
    # For streaming mode, implement stabilization
    if confidence < MIN_DETECTION_CONFIDENCE:
        # Low confidence, keep previous prediction if available
        if previous_detection and previous_detection != "No Sign":
            print(f"Low confidence ({confidence:.2f}), keeping previous: {previous_detection}")
            return previous_detection, hand_sign_id, 0.4
        else:
            print(f"Low confidence ({confidence:.2f}), no previous detection")
            return "No Sign", -1, 0.1
    
    # Add to recent predictions buffer
    recent_predictions.append((current_sign_text, confidence))
    
    # Count occurrences of each sign in the buffer
    prediction_counts = {}
    for sign, conf in recent_predictions:
        if sign not in prediction_counts:
            prediction_counts[sign] = {
                'count': 0,
                'total_confidence': 0
            }
        prediction_counts[sign]['count'] += 1
        prediction_counts[sign]['total_confidence'] += conf
    
    # Find the most frequent prediction
    most_frequent_sign = None
    max_count = 0
    for sign, data in prediction_counts.items():
        if data['count'] > max_count:
            max_count = data['count']
            most_frequent_sign = sign
    
    # Check if stable enough
    if most_frequent_sign and max_count >= PREDICTION_STABILITY_THRESHOLD:
        avg_confidence = prediction_counts[most_frequent_sign]['total_confidence'] / max_count
        # Clear buffer when we detect a stable sign
        print(f"Stable detection: {most_frequent_sign} (count: {max_count}, confidence: {avg_confidence:.2f})")
        recent_predictions.clear()
        return most_frequent_sign, hand_sign_id, float(avg_confidence)
    
    # Not stable enough, return previous or current with lower confidence
    if previous_detection and previous_detection != "No Sign":
        print(f"Not stable enough, keeping previous: {previous_detection}")
        return previous_detection, hand_sign_id, 0.4
    else:
        print(f"Not stable enough, using current with reduced confidence")
        return current_sign_text, hand_sign_id, confidence * 0.8

# Website routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/roadmap')
def roadmap():
    return render_template('roadmap.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

# REST API endpoints
@app.route('/api/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    if 'image' not in request.form:
        return jsonify({
            'success': False,
            'error': 'No image data provided'
        })
    
    # Streaming mode affects how we process results
    streaming_mode = request.form.get('streaming_mode', 'false').lower() == 'true'
    previous_detection = request.form.get('previous_detection')
    
    # Process the image
    try:
        # Decode base64 image
        image_data = request.form['image']
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to decode image: {str(e)}'
            })
        
        # Process for prediction
        landmark_list, handedness_info = process_image_for_prediction(image)
        
        if landmark_list is None:
            return jsonify({
                'success': True,
                'prediction': {
                    'sign_id': -1,
                    'sign_text': 'No Sign',
                    'confidence': 0.1,
                    'handedness': None
                },
                'processing_time': time.time() - start_time
            })
        
        # Make prediction
        sign_text, sign_id, confidence = predict_sign(landmark_list, previous_detection, streaming_mode)
        
        return jsonify({
            'success': True,
            'prediction': {
                'sign_id': int(sign_id),
                'sign_text': sign_text,
                'confidence': confidence,
                'handedness': handedness_info
            },
            'processing_time': time.time() - start_time
        })
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error processing image',
            'details': str(e),
            'processing_time': time.time() - start_time
        })

# WebSocket endpoints for real-time streaming
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    recent_predictions.clear()

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('stream_frame')
def handle_stream_frame(data):
    start_time = time.time()
    
    try:
        # Decode base64 image
        image_data = data.get('image', '')
        if not image_data or not isinstance(image_data, str):
            raise ValueError("Invalid image data format")
            
        # Handle different base64 formats
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Process the base64 image data
        try:
            # Ensure padding is correct for base64
            missing_padding = len(image_data) % 4
            if missing_padding:
                image_data += '=' * (4 - missing_padding)
                
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Some debugging info
            print(f"Received image: {image.format} {image.size[0]}x{image.size[1]}")
            
        except Exception as e:
            print(f"Error decoding image: {str(e)}")
            raise ValueError(f"Failed to decode image: {str(e)}")
        
        # Process for prediction
        landmark_list, handedness_info = process_image_for_prediction(image)
        
        if landmark_list is None:
            print("No hand landmarks detected")
            emit('prediction_result', {
                'success': True,
                'prediction': {
                    'sign_id': -1,
                    'sign_text': 'No Sign',
                    'confidence': 0.1,
                    'handedness': None
                },
                'processing_time': time.time() - start_time
            })
            return
        
        # Print landmark info for debugging
        print(f"Landmarks detected, length: {len(landmark_list)}")
        
        # Make prediction with temporal stabilization
        previous_detection = data.get('previous_detection')
        sign_text, sign_id, confidence = predict_sign(landmark_list, previous_detection, True)
        
        print(f"Prediction: {sign_text} (ID: {sign_id}) with confidence {confidence:.2f}")
        
        emit('prediction_result', {
            'success': True,
            'prediction': {
                'sign_id': int(sign_id),
                'sign_text': sign_text,
                'confidence': confidence,
                'handedness': handedness_info
            },
            'processing_time': time.time() - start_time
        })
        
    except Exception as e:
        print(f"Error processing stream frame: {str(e)}")
        import traceback
        traceback.print_exc()
        
        emit('prediction_result', {
            'success': False,
            'error': 'Error processing image',
            'details': str(e),
            'processing_time': time.time() - start_time
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': keypoint_classifier is not None,
        'mediapipe_loaded': hands is not None,
        'labels_loaded': len(labels)
    })

if __name__ == '__main__':
    print("Starting Sign Language Translator application...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)