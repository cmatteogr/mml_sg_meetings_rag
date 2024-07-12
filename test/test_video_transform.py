from unittest import TestCase
from src.media_transform.video_transform import extract_audio
import os


class Test(TestCase):
    def test_extract_audio(self):
        # Given the input arguments
        #video_filepath = r'../src/data/meetings_data/MML-SG Training meeting (2024-06-05 19_10 GMT-5).mp4'
        #video_audio_filepath = r'../src/data/meetings_data/MML-SG Training meeting (2024-06-05 19_10 GMT-5).mp3'
        folder_path = '../src/data/meetings_data/'
        mp4_files = [file for file in os.listdir(folder_path) if file.endswith('.mp4')]
        for mp4_file in mp4_files:
            print(f'Processing {mp4_file}')
            # When
            video_filepath = os.path.join(folder_path, mp4_file)
            video_audio_filepath = os.path.join(folder_path, mp4_file.replace('.mp4','.mp3'))
            extract_audio(video_filepath, video_audio_filepath)
