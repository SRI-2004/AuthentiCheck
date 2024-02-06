import cv2
import dlib
import numpy as np
import pyaudio
import wave
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize PyAudio for audio recording
p = pyaudio.PyAudio()

# Set up audio stream
chunk = 1024
rate = 44100
stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)

# Video capture (you can replace this with your own video capture implementation)
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Flag to check if audio is currently detected
audio_detected = False

# Buffer to store frames after audio detection
frame_buffer = []
buffer_duration = 2  # Duration to capture frames after audio detection (in seconds)
horizontal_threshold = 20
vertical_threshold = 3

amplitude_threshold = 5
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for facial landmark detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Read audio packet
    audio_data = np.frombuffer(stream.read(chunk), dtype=np.int16)

    # Check if audio is present
    if np.max(np.abs(audio_data)) > amplitude_threshold:
        # Get current timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Voice detected at {timestamp}")

        # Set the flag to indicate audio detection
        audio_detected = True

    # If audio is detected, add frames to the buffer
    if audio_detected:
        frame_buffer.append(frame)

    # Display the frame
    cv2.imshow("Lip Movement Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if the buffer duration has elapsed
    if audio_detected and len(frame_buffer) >= int(rate / chunk * buffer_duration):
        # Analyze lip movement during the buffered frames
        # Implement your lip movement analysis logic here using the frames in 'frame_buffer'
        for face in faces:
            # Predict facial landmarks
            landmarks = predictor(gray, face)

            # Extract lip landmarks (assuming landmarks 48 to 67 represent the outer lip)
            lip_landmarks = landmarks.parts()[48:68]

            # Draw the lip landmarks on the frame
            for point in lip_landmarks:
                cv2.circle(frame, (point.x, point.y), 1, (0, 0, 255), -1)

            # Calculate lip movement
            lip_horizontal_movement = lip_landmarks[-1].x - lip_landmarks[0].x
            lip_vertical_movement = lip_landmarks[-1].y - lip_landmarks[0].y

            # Add your lip movement analysis logic here
            if abs(lip_horizontal_movement) > horizontal_threshold and abs(lip_vertical_movement) > vertical_threshold:
                times = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"Lip movement detected! {times}")
                print("lip sync verified")
            else:
                times = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"Lip movement not detected {times}")


        # Reset flags and buffer
        audio_detected = False
        frame_buffer = []

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()

# Stop the audio stream and PyAudio
stream.stop_stream()
stream.close()
p.terminate()