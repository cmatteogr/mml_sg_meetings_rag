from unittest import TestCase
from src.media_transform.video_transform import extract_audio
import os


class Test(TestCase):
    def test_extract_audio(self):
        # Given the input arguments
        video_filepath = r'../src/data/meetings_data/MML-SG Training meeting (2024-06-05 19_10 GMT-5).mp4'
        video_audio_filepath = r'../src/data/meetings_data/MML-SG Training meeting (2024-06-05 19_10 GMT-5).mp3'
        # When
        extract_audio(video_filepath, video_audio_filepath)
        # Then
        self.assertTrue(os.path.exists(video_audio_filepath), 'Video audio file not generated')
