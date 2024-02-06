import librosa
import numpy as np
from scipy.spatial.distance import cosine

class AudioVerification:
    def __init__(self, audio_file1, audio_file2):
        self.features1 = self.extract_features(audio_file1)
        self.features2 = self.extract_features(audio_file2)

    def verify(self, threshold=0.9):
        similarity_score = self.calculate_similarity(self.features1, self.features2)
        print("similarity score",similarity_score)
        if similarity_score >= threshold:

            return 1  # Similar audio files
        elif similarity_score<threshold:
            return 2  # Different audio files
        else:
            return 0

    def extract_features(self, audio_file):
        y, sr = librosa.load(audio_file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)  # Reduced to 13 coefficients
        mfcc_mean = np.mean(mfcc, axis=1)
        features = mfcc_mean

        return features

    def calculate_similarity(self, feature_vector1, feature_vector2):
        return 1 - cosine(feature_vector1, feature_vector2)
