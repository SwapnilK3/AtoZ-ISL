document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const startBtn = document.getElementById('startBtn');
    const captureBtn = document.getElementById('captureBtn');
    const stopBtn = document.getElementById('stopBtn');
    const resultContainer = document.getElementById('result');
    const confidenceMeter = document.querySelector('.meter-fill');
    const confidenceValue = document.getElementById('confidence-value');
    const processingTime = document.querySelector('.processing-time span');
    const streamingModeToggle = document.getElementById('streamingMode');
    const loadingIndicator = document.getElementById('loading-indicator');
    const historyContainer = document.getElementById('history-container');
    
    // Variables
    let stream = null;
    let isStreamingMode = false;
    let isProcessing = false;
    let lastDetectedSign = null;
    let detectionHistory = [];
    let socket = null;
    let frameSkip = 3;
    let frameCount = 0;
    const MAX_HISTORY_ITEMS = 5;
    
    // Event Listeners
    startBtn.addEventListener('click', startCamera);
    captureBtn.addEventListener('click', captureImage);
    stopBtn.addEventListener('click', stopCamera);
    streamingModeToggle.addEventListener('change', toggleStreamingMode);
    
    // Initialize Socket.IO
    function initSocketIO() {
        if (!socket) {
            socket = io.connect(window.location.origin);
            
            socket.on('connect', function() {
                console.log('WebSocket connected');
            });
            
            socket.on('disconnect', function() {
                console.log('WebSocket disconnected');
            });
            
            socket.on('prediction_result', function(data) {
                handlePredictionResult(data);
                isProcessing = false;
                
                if (isStreamingMode && stream) {
                    // Continue streaming with a delay to avoid overwhelming the server
                    setTimeout(captureAndStreamFrame, 100);
                }
            });
            
            socket.on('connect_error', function(error) {
                console.error('Socket.IO connection error:', error);
                showError('Connection error. Please refresh and try again.');
            });
        }
    }
    
    // Functions
    async function startCamera() {
        try {
            // Request camera access with explicit constraints
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
            
            // Set stream to video element
            video.srcObject = stream;
            
            // Wait for video to be loaded before initializing
            video.onloadedmetadata = function() {
                console.log(`Video dimensions: ${video.videoWidth}x${video.videoHeight}`);
                
                // Initialize canvas with correct dimensions
                const ctx = canvas.getContext('2d');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                
                // Clear canvas
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Update UI
                toggleButtonState(true);
                
                // Initialize WebSocket for streaming
                initSocketIO();
                
                // Enable streaming mode if toggle is checked
                if (streamingModeToggle.checked) {
                    startStreamingMode();
                }
            };
            
            // Handle video errors
            video.onerror = function(err) {
                console.error('Video error:', err);
                showError('Error accessing camera stream');
            };
            
        } catch (err) {
            console.error('Error accessing camera:', err);
            showError('Could not access camera. Please ensure camera permissions are enabled.');
        }
    }
    
    function toggleStreamingMode() {
        isStreamingMode = streamingModeToggle.checked;
        
        if (stream) {
            if (isStreamingMode) {
                startStreamingMode();
            } else {
                stopStreamingMode();
            }
        }
    }
    
    function startStreamingMode() {
        isStreamingMode = true;
        captureBtn.disabled = true;
        frameCount = 0;
        
        // Start the streaming process
        if (!isProcessing) {
            captureAndStreamFrame();
        }
    }
    
    function stopStreamingMode() {
        isStreamingMode = false;
        captureBtn.disabled = false;
    }
    
    function captureImage() {
        if (!stream || isProcessing) return;
        
        captureAndProcessFrame(false);
    }
    
    function captureAndStreamFrame() {
        if (!stream || !isStreamingMode) return;
        if (isProcessing) return;
        
        frameCount++;
        
        // Skip frames to reduce load (process every nth frame)
        if (frameCount < frameSkip) {
            setTimeout(captureAndStreamFrame, 33); // ~30fps
            return;
        }
        
        frameCount = 0;
        isProcessing = true;
        
        try {
            // Ensure video is playing
            if (video.paused || video.ended) {
                isProcessing = false;
                return;
            }
            
            // Get current dimensions
            const width = video.videoWidth;
            const height = video.videoHeight;
            
            // Check if dimensions are valid
            if (!width || !height) {
                console.error('Invalid video dimensions:', width, height);
                isProcessing = false;
                setTimeout(captureAndStreamFrame, 500);
                return;
            }
            
            // Ensure canvas matches video dimensions
            if (canvas.width !== width || canvas.height !== height) {
                canvas.width = width;
                canvas.height = height;
            }
            
            // Draw current frame to canvas with horizontal flip to correct mirroring
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, width, height);
            
            // Horizontally flip the image (un-mirror it)
            ctx.save();
            ctx.translate(width, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(video, 0, 0, width, height);
            ctx.restore();
            
            // Get image data as base64
            const imageData = canvas.toDataURL('image/jpeg', 0.9);
            
            // Verify data is present
            if (!imageData || imageData === 'data:,') {
                console.error('Failed to capture image data');
                isProcessing = false;
                setTimeout(captureAndStreamFrame, 500);
                return;
            }
            
            // Send via WebSocket
            socket.emit('stream_frame', {
                image: imageData,
                previous_detection: lastDetectedSign,
                timestamp: Date.now(),
                unmirrored: true  // Flag to indicate the image is already unmirrored
            });
            
        } catch (error) {
            console.error('Error capturing frame:', error);
            isProcessing = false;
            setTimeout(captureAndStreamFrame, 1000);
        }
    }

    function captureAndProcessFrame(isStreaming = false) {
        if (isProcessing) return;
        
        isProcessing = true;
        showLoading(true);
        
        try {
            // Ensure video is playing
            if (video.paused || video.ended) {
                isProcessing = false;
                showLoading(false);
                return;
            }
            
            // Get current dimensions
            const width = video.videoWidth;
            const height = video.videoHeight;
            
            // Ensure canvas matches video dimensions
            canvas.width = width;
            canvas.height = height;
            
            // Draw current frame to canvas with horizontal flip
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, width, height);
            
            // Horizontally flip the image (un-mirror it)
            ctx.save();
            ctx.translate(width, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(video, 0, 0, width, height);
            ctx.restore();
            
            // Get image data as base64
            const imageData = canvas.toDataURL('image/jpeg', 0.9);
            
            // Process via HTTP API for single captures
            processImage(imageData, isStreaming);
            
        } catch (error) {
            console.error('Error capturing frame:', error);
            isProcessing = false;
            showLoading(false);
            showError('Error capturing image');
        }
    }
    
    async function processImage(imageData, isStreaming = false) {
        try {
            const formData = new FormData();
            formData.append('image', imageData);
            formData.append('streaming_mode', isStreaming);
            
            if (lastDetectedSign) {
                formData.append('previous_detection', lastDetectedSign);
            }
            
            const startTime = performance.now();
            
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            handlePredictionResult(data, performance.now() - startTime);
            
        } catch (error) {
            console.error('Error processing image:', error);
            displayResult('Connection error. Please try again.', true);
            updateConfidence(0);
        } finally {
            isProcessing = false;
            showLoading(false);
        }
    }
    
    function stopCamera() {
        stopStreamingMode();
        
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
            video.srcObject = null;
            toggleButtonState(false);
            
            // Reset results
            resetResults();
        }
    }
    
    function toggleButtonState(isStarted) {
        startBtn.disabled = isStarted;
        captureBtn.disabled = !isStarted || isStreamingMode;
        stopBtn.disabled = !isStarted;
    }
    
    function handlePredictionResult(data, clientTime = 0) {
        showLoading(false);
        
        if (data.success) {
            const prediction = data.prediction;
            const signText = prediction.sign_text;
            const confidence = prediction.confidence || 0.5;
            
            // Update UI
            displayResult(signText, false, confidence < 0.4);
            updateConfidence(confidence * 100);
            updateProcessingTime(data.processing_time, clientTime);
            
            // Keep track of detection for streaming mode
            if (signText !== 'No Sign' && confidence > 0.4) {
                // Only update if different from last detected sign and above threshold
                if (signText !== lastDetectedSign) {
                    lastDetectedSign = signText;
                    addToHistory(signText, confidence);
                }
            }
        } else {
            displayResult(data.error || 'Error processing image', true);
            updateConfidence(0);
        }
    }
    
    function displayResult(text, isError = false, isLowConfidence = false) {
        if (isError) {
            resultContainer.innerHTML = `<div class="error-message">${text}</div>`;
        } else if (isLowConfidence) {
            resultContainer.innerHTML = `<div class="result-recognized low-confidence">${text}</div>`;
        } else if (text === 'No Sign') {
            resultContainer.innerHTML = `<div class="result-recognized faded">Waiting for sign...</div>`;
        } else {
            resultContainer.innerHTML = `<div class="result-recognized">${text}</div>`;
        }
    }
    
    function updateConfidence(value) {
        confidenceMeter.style.width = `${value}%`;
        confidenceValue.textContent = `${Math.round(value)}%`;
        
        // Change color based on confidence
        if (value > 75) {
            confidenceMeter.style.backgroundColor = '#10b981'; // Green
        } else if (value > 50) {
            confidenceMeter.style.backgroundColor = '#f59e0b'; // Yellow
        } else {
            confidenceMeter.style.backgroundColor = '#ef4444'; // Red
        }
    }
    
    function updateProcessingTime(serverTime, clientTime) {
        const total = serverTime + (clientTime / 1000);
        processingTime.textContent = `${Math.round(total * 1000)} ms`;
    }
    
    function showError(message) {
        resultContainer.innerHTML = `<div class="error-message">${message}</div>`;
        updateConfidence(0);
    }
    
    function resetResults() {
        resultContainer.innerHTML = `<p class="placeholder-text">Translation will appear here...</p>`;
        confidenceMeter.style.width = '0%';
        confidenceValue.textContent = '0%';
        processingTime.textContent = '0 ms';
        historyContainer.innerHTML = '';
        detectionHistory = [];
        lastDetectedSign = null;
    }
    
    function showLoading(show) {
        if (loadingIndicator) {
            loadingIndicator.style.display = show ? 'flex' : 'none';
        }
    }
    
    function addToHistory(sign, confidence) {
        // Add to history array
        detectionHistory.unshift({
            sign,
            confidence,
            timestamp: new Date()
        });
        
        // Limit history size
        if (detectionHistory.length > MAX_HISTORY_ITEMS) {
            detectionHistory.pop();
        }
        
        // Update UI
        updateHistoryUI();
    }
    
    function updateHistoryUI() {
        if (!historyContainer) return;
        
        historyContainer.innerHTML = '';
        
        if (detectionHistory.length === 0) {
            historyContainer.innerHTML = '<p class="placeholder-text">No detections yet</p>';
            return;
        }
        
        detectionHistory.forEach(item => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            
            const time = item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            
            historyItem.innerHTML = `
                <span class="history-sign">${item.sign}</span>
                <div class="history-details">
                    <span class="history-confidence">${Math.round(item.confidence * 100)}%</span>
                    <span class="history-time">${time}</span>
                </div>
            `;
            
            historyContainer.appendChild(historyItem);
        });
    }
});