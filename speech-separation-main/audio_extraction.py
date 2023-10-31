from moviepy.editor import *

video_path = input("enter video path")
video_file = video_path
audio_file = "audio.wav"

# Load video file
video = VideoFileClip(video_file)

# Extract audio
audio = video.audio

# Write audio to .wav file
audio.write_audiofile(audio_file)
