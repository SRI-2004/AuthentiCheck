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
import sounddevice as sd
class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Create OpenCV video capture
        self.cap = cv2.VideoCapture(0)

        # Set up audio parameters
        self.audio_channels = 1  # Mono audio
        self.audio_samplerate = 44100  # Audio sample rate
        self.audio_frames_per_buffer = 1024  # Number of frames per buffer

        # Create buttons for face and audio verification
        self.face_button = ttk.Button(window, text="Face Verification", command=self.verify_face)
        self.face_button.pack(pady=10)
        self.audio_frame = 0
        self.audio_button = ttk.Button(window, text="Audio Verification",
                                       command=lambda: self.verify_audio(self.audio_frame))
        self.audio_button.pack(pady=10)
        self.audio_verification_is_done = False
        # Create a flag for video and audio thread to stop
        self.is_capturing = True

        self.face_verified = False
        self.audio_verified = False
        # Start video capture thread
        self.video_thread = threading.Thread(target=self.capture_video)
        self.video_thread.start()

        # Start audio capture thread
        self.audio_thread = threading.Thread(target=self.capture_audio)
        self.audio_thread.start()

        self.verify_environment_flag = False

        # Start environment verification thread
        self.environment_thread = threading.Thread(target=self.verify_environment_thread)
        self.environment_thread.start()

        self.is_tracking = False

        # Start eyesight tracker thread
        self.eyesight_tracker_thread = threading.Thread(target=self.verify_eyesight_thread)
        self.eyesight_tracker_thread.start()
        # Close the app properly when the window is closed
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)


    def verify_eyesight_thread(self):
        while self.is_capturing:
            if self.is_tracking:
                ret, frame = self.cap.read()
                if ret:
                    self.track_eyesight(frame)
            time.sleep(1)
    def track_eyesight(self,frame):
        eyesight_tracker = EyesightTracker()

        result = eyesight_tracker.track(frame)
        if (result == 1):
            print("Eyesight out of frame")

    def verify_environment_thread(self):
        while self.is_capturing:
            if self.verify_environment_flag:
                ret, frame = self.cap.read()
                if ret:
                    self.verify_environment(frame)

    def verify_environment(self,frame):
        # Create an instance of EnvironmentDetection
        # Convert the frame to a NumPy array with the correct data type

        environment_detection = EnvironmentDetection()

        result = environment_detection.verify(frame=frame)

        if result == 1:
            print(f"More than one person detected: ")
        elif result == 2:
            print("Technological product like a phone detected.")
        else:
            print("No objects detected.")


    def verify_face(self):
        ret, frame = self.cap.read()
        if ret:
            face_verification = FaceVerification()
            result = face_verification.verify(frame)

            if result == 1:
                self.face_button.config(text="Verified", style="Verified.TButton")
                self.face_verified = True
            elif result == 2:
                self.face_button.config(text="More than one human detected",style="Error.TButton")
            else:
                self.face_button.config(text="Error, please try again", style="Error.TButton")

    def verify_audio(self,audio_frame):
        temporary_audio_file = "temporary_audio_file.wav"
        write(temporary_audio_file, data=audio_frame,rate=self.audio_samplerate)

        # Create an instance of AudioVerification
        audio_verification = AudioVerification(temporary_audio_file,audio_frame)

        # Verify the similarity with the temporary audio file
        result = audio_verification.verify()
        if result == 1:
            self.audio_button.config(text="Error, please try again", style="Error.TButton")
        elif result==2:
            self.audio_button.config(text="Verified", style="Verified.TButton")
            self.audio_verified = True
        self.audio_verification_is_done = True
    def capture_video(self):
        while self.is_capturing:
            ret, frame = self.cap.read()
            if not ret:
                break
            # print(ret)
            # print(frame)
            # Display the captured frame in the OpenCV window
            cv2.imshow("Video", frame)
            if self.face_verified and self.audio_verified:
                # Set the flag to verify environment in a separate thread
                self.face_button.pack_forget()
                self.audio_button.pack_forget()
                self.verify_environment_flag = True
                self.is_tracking = True
                # If both are verified, hide the buttons

            # Break the loop if the user closes the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture when the thread ends
        self.cap.release()
        cv2.destroyAllWindows()

    def capture_audio(self):
        with sd.InputStream(channels=self.audio_channels,
                            samplerate=self.audio_samplerate,
                            blocksize=self.audio_frames_per_buffer,
                            dtype=np.int16):
            while self.is_capturing and not self.audio_verification_is_done:
                audio_frame = sd.rec(frames=self.audio_frames_per_buffer, channels=self.audio_channels, dtype='int16')
                sd.wait()  # Wait for the recording to complete
                self.audio_frame = audio_frame
                self.verify_audio(audio_frame)

    def on_close(self):
        # Stop video capture thread
        self.is_capturing = False
        self.video_thread.join()
        self.audio_thread.join()
        self.eyesight_tracker_thread.join()
        self.environment_thread.join()
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
