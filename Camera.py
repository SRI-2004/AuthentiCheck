import queue
import threading
import time
import cv2
import tkinter as tk
from tkinter import ttk
from scipy.io.wavfile import write
from environment_check import EnvironmentDetection
from eyesight_tracker import EyesightTracker
from face_verification import FaceVerification
from audio_verification import AudioVerification
import numpy as np
import pyaudio

from lip_detection import LipDetection


class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.eyesight_tracker = EyesightTracker()
        # Create OpenCV video capture
        self.cap = cv2.VideoCapture(0)
        self.audio_thread_stop = threading.Event()
        # Set up audio parameters
        self.audio_channels = 1  # Mono audio
        self.audio_samplerate = 44100  # Audio sample rate
        self.audio_frames_per_buffer = 1024  # Number of frames per buffer

        # Create buttons for face and audio verification
        self.face_button = ttk.Button(window, text="Face Verification", command=self.verify_face)
        self.face_button.pack(pady=10)
        self.audio_frame = np.zeros(self.audio_frames_per_buffer, dtype=np.int16)

        self.audio_button = ttk.Button(window, text="Audio Verification",
                                       command= self.start_audio_thread)
        self.audio_button.pack(pady=10)
        self.audio_capture_duration = 3
        self.audio_verification_is_done = False
        self.environment = EnvironmentDetection()
        # PyAudio instance for audio recording
        self.p = pyaudio.PyAudio()
        # Create a flag for video and audio thread to stop
        self.shared_frame_buffer = queue.Queue(maxsize=1)
        self.is_capturing = True
        # self.env_lock = threading.Lock()
        # self.lip_lock = threading.Lock()
        # self.eye_lock = threading.Lock()
        self.face_verified = True
        self.audio_verified = True

        # Start video capture thread
        self.video_thread = threading.Thread(target=self.capture_video)
        self.video_thread.start()

        # Start audio capture thread
        self.audio_thread = threading.Thread(target=self.capture_audio)
        # self.audio_thread.start()
        self.face_verification = FaceVerification()
        self.verify_environment_flag = False

        # Start environment verification thread
        self.environment_thread = threading.Thread(target=self.verify_environment_thread)
        self.environment_thread.start()

        self.is_tracking = False

        # Start eyesight tracker thread
        self.eyesight_tracker_thread = threading.Thread(target=self.verify_eyesight_thread)
        self.eyesight_tracker_thread.start()

        self.is_syncing = False
        self.lip_sync_frames = []
        # Start eyesight tracker thread
        self.lip_sync_thread = threading.Thread(target=self.verify_lip_sync_thread)
        self.lip_sync_thread.start()

        # Close the app properly when the window is closed
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def verify_lip_sync_thread(self):
        print("Lip sync thread started")
        while self.is_capturing:
            if self.is_syncing:
                # Call capture_audio_frames to get the frame buffer
                rate = 44100
                chunk = 1024
                buffer_duration = 2
                frame_buffer = self.capture_audio_frames()

                # Perform lip sync verification using a LipDetection class
                lip_detection = LipDetection()  # Assuming LipDetection is the class with the verify method
                if len(frame_buffer) >= 24:
                    # Verify lip sync using the frame buffer

                    if lip_detection.verify(frame_buffer):
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        print(f"Voice detected at {timestamp}")
                        print("Lip sync verified")
                    else:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        print(f"Voice detected at {timestamp}")
                        print("No lip sync detected")


    def capture_audio_frames(self):
        # Set up audio stream
        chunk = 1024
        rate = 44100
        stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)

        # Buffer to store frames after audio detection
        self.lip_sync_frames = []
        buffer_duration = 2  # Duration to capture frames after audio detection (in seconds)
        amplitude_threshold = 5

        # Flag to check if audio is currently detected
        audio_detected = False
        start_time = time.time()

        while time.time() - start_time < buffer_duration:
            audio_data = np.frombuffer(stream.read(chunk), dtype=np.int16)

            # Check if audio is present
            if np.max(np.abs(audio_data)) > amplitude_threshold:
                # Get current timestamp
                # timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                # print(f"Voice detected at {timestamp}")

                # Set the flag to indicate audio detection
                audio_detected = True

            # If audio is detected, add frames to the buffer
            if audio_detected:
                frame = self.shared_frame_buffer.get()
                self.lip_sync_frames.append(frame)

                # Display the frame if needed

        # Release the audio stream
        frame_buffer = self.lip_sync_frames
        stream.stop_stream()
        stream.close()
        return frame_buffer

    def verify_eyesight_thread(self):
        frame_counter = 0
        while self.is_capturing:
            time.sleep(3)
            if self.is_tracking:
                frame = self.shared_frame_buffer.get()
                print(frame_counter)
                self.track_eyesight(frame)
                frame_counter += 1

    def track_eyesight(self, frame):

        result = self.eyesight_tracker.track(frame)
        if result == None:
            print("Eyesight out of frame")

    def verify_environment_thread(self):
        frame_counter = 0
        while self.is_capturing:
            if self.verify_environment_flag:
                if frame_counter % 15 == 0:
                    frame = self.shared_frame_buffer.get()
                    self.verify_environment(frame)
                frame_counter += 1
    def verify_environment(self, frame):
        # Create an instance of EnvironmentDetection
        # Convert the frame to a NumPy array with the correct data type

        result = self.environment.verify(frame=frame)

        if result == 2 :
            print("More than one person detected: ")
        elif result == 1 or result == 3:
            print("Environment change detected")
        else:
            print("No objects detected.")

    def verify_face(self):
        print("Entering verify face")
        frame = self.shared_frame_buffer.get()
        if True:
            result = self.face_verification.verify(frame)

            if result == 1:
                self.face_button.config(text="Verified", style="Verified.TButton")
                self.face_verified = True
            elif result == 2:
                self.face_button.config(text="More than one human detected", style="Error.TButton")
            else:
                self.face_button.config(text="Error, please try again", style="Error.TButton")

    def verify_audio(self, audio_frame):
        temporary_audio_file = "temporary_audio_file.wav"
        write(temporary_audio_file, data=audio_frame, rate=self.audio_samplerate)

        # Create an instance of AudioVerification
        audio_verification = AudioVerification(temporary_audio_file, "test_audio.mp3")

        # Verify the similarity with the temporary audio file
        result = audio_verification.verify()
        if result == 2:
            self.audio_button.config(text="Error, please try again", style="Error.TButton")
            print("mismatch")
        elif result == 1:
            self.audio_button.config(text="Verified", style="Verified.TButton")
            self.audio_verified = True
        else:
            self.audio_button.config(text="Error, please try again", style="Error.TButton")
            print("error")
        self.audio_verification_is_done = True
        self.audio_thread_stop.set()

    def capture_video(self):
        frame_counter = 0
        while self.is_capturing:
            ret, frame = self.cap.read()
            if not ret:
                break
            # if frame_counter % 12 == 0:
            #     frame = self.shared_frame_buffer.get()
            #     cv2.imshow("Video", frame)
            # frame_counter +=  1
            if not self.shared_frame_buffer.empty():
                self.shared_frame_buffer.get()
            self.shared_frame_buffer.put(frame)


            if self.face_verified and self.audio_verified:
                # Set the flag to verify environment in a separate thread
                self.face_button.pack_forget()
                self.audio_button.pack_forget()
                self.is_tracking = True
                self.verify_environment_flag = False
                self.is_syncing = False

                # If both are verified, hide the buttons

            # Break the loop if the user closes the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture when the thread ends
        self.cap.release()
        cv2.destroyAllWindows()

    def capture_audio_frame(self, start_time):
        stream = self.p.open(format=pyaudio.paInt16,
                             channels=self.audio_channels,
                             rate=self.audio_samplerate,
                             input=True,
                             frames_per_buffer=self.audio_frames_per_buffer)

        audio_frames = []
        elapsed_time = 0

        while elapsed_time < self.audio_capture_duration:
            audio_frame = stream.read(self.audio_frames_per_buffer)
            audio_frame = np.frombuffer(audio_frame, dtype=np.int16)
            audio_frames.append(audio_frame)

            elapsed_time = time.time() - start_time

        stream.stop_stream()
        stream.close()

        # Concatenate the audio frames into a single array
        audio_frame = np.concatenate(audio_frames)

        return audio_frame

    def capture_audio(self):
        start_time = time.time()

        while self.is_capturing and not self.audio_thread_stop.is_set():
            audio_frame = self.capture_audio_frame(start_time)
            print(audio_frame)
            self.verify_audio(audio_frame)

    def start_audio_thread(self):
        # Disable the button to prevent multiple thread starts
        self.audio_button["state"] = "disabled"

        # Start the audio thread
        self.audio_thread = threading.Thread(target=self.capture_audio)
        self.audio_thread.start()
    def on_close(self):
        # Stop video capture thread
        self.is_capturing = False
        self.video_thread.join()
        self.audio_thread.join()
        self.eyesight_tracker_thread.join()
        self.environment_thread.join()

        # Terminate PyAudio instance
        self.p.terminate()

        self.window.destroy()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        style = ttk.Style()
        style.configure("Verified.TButton", background="green")
        style.configure("Error.TButton", foreground="red")
        app = CameraApp(root, "Video/Audio Capture App")
        root.mainloop()
    except Exception as e:
        print(f"An error occurred: {e}")