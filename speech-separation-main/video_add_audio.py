from moviepy.editor import *

# Load the video and audio files
video = VideoFileClip(
    "/content/lip-reading-deeplearning/results/output_video.mp4")
audio = AudioFileClip("/content/mywav_reduced_noise.wav")

# Set the audio to the same duration as the video
audio = audio.set_duration(video.duration)

# Mix the audio with the video
final_clip = video.set_audio(audio)

# Write the final clip to a new file
final_clip.write_videofile("output1.mp4")
