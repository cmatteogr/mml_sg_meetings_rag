"""
Video Transformations

"""
# Import libraries
from moviepy.editor import VideoFileClip
import os


def extract_audio(video_filepath: str, video_audio_filepath: str):
    """
    Extract audio from video file
    :param video_filepath:
    :param video_audio_filepath:
    :return:
    """
    # Validate filepaths
    assert os.path.exists(video_filepath), 'Video filepath does not exist'
    assert os.path.splitext(video_filepath)[1] == '.mp4', 'Invalid video extension, it must be mp4'
    assert os.path.splitext(video_audio_filepath)[1] == '.mp3', 'Invalid video audio extension, it must be mp3'

    # Load the video file
    video = VideoFileClip(video_filepath)

    # Extract the audio
    audio = video.audio

    # Save the audio to a file
    audio.write_audiofile(video_audio_filepath)
