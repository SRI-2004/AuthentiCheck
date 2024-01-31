import librosa
import numpy as np
from scipy.spatial.distance import cosine


class AudioVerification:
    def __init__(self,audio_file1, audio_file2):
        self.audio_file1 = audio_file1  # Change this to the actual path
        self.audio_file2 = audio_file2

    def verify(self, threshold=0.9):
        # Extract features from both audio files
        features1 = self.extract_features(self.audio_file1)
        features2 = self.extract_features(self.audio_file2)

        # Calculate similarity score
        similarity_score = self.calculate_similarity(features1, features2)

        # Compare with the threshold
        if similarity_score >= threshold:
            return 1  # Return 1 for similar audio files
        elif similarity_score<threshold:
            return 2  # Return 0 for different audio files
        else:
            return 0

    def extract_features(self, audio_file):
        # Extract MFCC features
        y, sr = librosa.load(audio_file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        # Additional feature: calculate the mean and standard deviation of MFCC values
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # Concatenate mean and standard deviation features
        features = np.concatenate([mfcc_mean, mfcc_std])

        return features

    def calculate_similarity(self, feature_vector1, feature_vector2):
        # Calculate cosine similarity between feature vectors
        return 1 - cosine(feature_vector1, feature_vector2)
