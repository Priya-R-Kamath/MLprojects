from moviepy.editor import *
import speechbrain as sb
import torchaudio
from IPython.display import Audio
from speechbrain.dataio.dataio import read_audio
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.pretrained import SepformerSeparation as separator
from scipy.io import wavfile
from scipy.io import wavfile
import noisereduce as nr

video_path = input("enter video path")
video_file = video_path
audio_file = "audio.wav"

# Load video file
video = VideoFileClip(video_file)

# Extract audio
audio = video.audio

# Write audio to .wav file
audio.write_audiofile(audio_file)


model = separator.from_hparams(
    source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')

model_path = "audio.wav"
est_sources = model.separate_file(path=model_path)

wavfile.write('output1.wav', rate=8000,
              data=est_sources[:, :, 0].detach().cpu().squeeze().numpy())
Audio(est_sources[:, :, 0].detach().cpu().squeeze(), rate=8000)

wavfile.write('output2.wav', rate=8000,
              data=est_sources[:, :, 1].detach().cpu().squeeze().numpy())
Audio(est_sources[:, :, 1].detach().cpu().squeeze(), rate=8000)


# load data
rate, data = wavfile.read("output1.wav")
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write("mywav_reduced_noise1.wav", rate, reduced_noise)

# load data
rate, data = wavfile.read("output2.wav")
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write("mywav_reduced_noise2.wav", rate, reduced_noise)


# Load the video and audio files
video = VideoFileClip(video_path)
audio = AudioFileClip("mywav_reduced_noise1.wav")

# Set the audio to the same duration as the video
audio = audio.set_duration(video.duration)

# Mix the audio with the video
final_clip = video.set_audio(audio)

# Write the final clip to a new file
final_clip.write_videofile("output1.mp4")


video = VideoFileClip(video_path)
audio = AudioFileClip("mywav_reduced_noise2.wav")

# Set the audio to the same duration as the video
audio = audio.set_duration(video.duration)

# Mix the audio with the video
final_clip = video.set_audio(audio)

# Write the final clip to a new file
final_clip.write_videofile("output2.mp4")
