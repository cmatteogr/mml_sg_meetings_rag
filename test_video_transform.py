from unittest import TestCase
from src.media_transform.video_transform import extract_audio
import os


class Test(TestCase):
    def test_extract_audio(self):
        # Given the input arguments

        extract_audio(video_filepath, video_audio_filepath)

        self.assertTrue(os.path.exists(video_audio_filepath), 'Video audio file not generated')
