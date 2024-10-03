from unittest import TestCase
from src.media_transform.text_formatter import preprocess_recording_transcription
import pandas as pd


class Test(TestCase):
    def test_preprocess_recording_transcription(self):
        # Given
        recording_transcrip_filepath = '../src/data/meetings_data/MML-SG Training meeting (2024-08-14 19_11 GMT-5).csv'
        recording_transcrip_df = pd.read_csv(recording_transcrip_filepath)
        # When
        recording_transcr_df = preprocess_recording_transcription(recording_transcrip_df)
        recording_transcr_df.to_json('../src/data/meetings_data/MML-SG Training meeting (2024-06-05 19_10 GMT-5).json')
        # Then
