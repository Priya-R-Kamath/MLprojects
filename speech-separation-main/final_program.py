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
import cv2
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from python_speech_features import mfcc
from google.colab.patches import cv2_imshow


video_path = input("enter video path")
video_file = video_path
audio_file = "audio.wav"

# Load video file
video = VideoFileClip(video_file)

# Extract audio
audio = video.audio

# Write audio to .wav file
audio.write_audiofile(audio_file)

print('Speech Processign....')
model = separator.from_hparams(
    source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')

model_path = "audio.wav"
est_sources = model.separate_file(path=model_path)

print("Speech separation....")
wavfile.write('output1.wav', rate=8000,
              data=est_sources[:, :, 0].detach().cpu().squeeze().numpy())
Audio(est_sources[:, :, 0].detach().cpu().squeeze(), rate=8000)

wavfile.write('output2.wav', rate=8000,
              data=est_sources[:, :, 1].detach().cpu().squeeze().numpy())
Audio(est_sources[:, :, 1].detach().cpu().squeeze(), rate=8000)

print("Noice reduction....")
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


# Define the number of speakers
num_speakers = 2

# Load the preprocessed audio data for each speaker
speaker1_data, sr = librosa.load('output1.wav', sr=16000)
speaker2_data, sr = librosa.load('output2.wav', sr=16000)

print("Speach separation done...")
print("starting with speaker identification.....")
print("Extracting audio features")

# Extract the features from each audio signal
speaker1_features = mfcc(speaker1_data, samplerate=sr, numcep=24)
speaker2_features = mfcc(speaker2_data, samplerate=sr, numcep=24)

# Concatenate the features into a single matrix
features = np.concatenate((speaker1_features, speaker2_features))

# Train a Gaussian mixture model (GMM) for each speaker
gmm1 = GaussianMixture(n_components=16, covariance_type='diag')
gmm1.fit(speaker1_features)

gmm2 = GaussianMixture(n_components=16, covariance_type='diag')
gmm2.fit(speaker2_features)

# Load the video file
video_capture = cv2.VideoCapture(video_path)

# Define the face detection algorithm
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize counts for left and right faces
face1 = 0
face2 = 0


audio1_path = 'mywav_reduced_noise1.wav'
audio2_path = 'mywav_reduced_noise2.wav'

print("Matching audio with video")
# Loop over each frame in the video
while True:
    # Read a frame from the video file
    ret, frame = video_capture.read()

    # If the video file has ended, break out of the loop
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    # Reset left and right face counts for each frame

    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Calculate the midpoint of the face
        face_midpoint = x + (w // 2)

        # Extract the face region from the frame
        face = frame[y:y+h, x:x+w]

        # Convert the face region to grayscale for speaker recognition
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Extract the MFCC features from the face region
        face_features = mfcc(face_gray, samplerate=sr, numcep=24)

        # Predict the speaker for the face region using the GMMs
        likelihood1 = np.sum(gmm1.score_samples(face_features))
        likelihood2 = np.sum(gmm2.score_samples(face_features))

        # Assign a label to the speaker based on the likelihood
        if likelihood1 > likelihood2:
            speaker_label = 1
        else:
            speaker_label = 2

        # Increment left or right face count based on speaker label and face position
        if speaker_label == 1:
            if face_midpoint <= frame.shape[1] // 2:
                face1 += 1
            else:
                face2 += 1

        # Draw a rectangle around the face based on speaker label and face position
        if speaker_label == 1:
            if face_midpoint <= frame.shape[1] // 2:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            if face_midpoint > frame.shape[1] // 2:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with the detected faces
    # cv2_imshow(frame)

    # Wait for a key press to move to the next frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print the final counts of left and right faces


# Release the video capture object and close the display window
video_capture.release()
cv2.destroyAllWindows()

print("Writing Finaloutput files...")
if (face1 < face2):

    # Create a VideoCapture object to read the video
    cap = cv2.VideoCapture(video_path)

    # Define the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Variables to store the dimensions of the video frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output1.mp4', fourcc, 30,
                          (frame_width, frame_height))

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 2:
            (x1, y1, w1, h1) = faces[0]
            (x2, y2, w2, h2) = faces[1]
            if x1 > x2:
                (x1, y1, w1, h1) = (x2, y2, w2, h2)

            # Draw a rectangle around the person on the right
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

        # Write the output frame to the output video file
        out.write(frame)

        # Display the resulting frame
        # cv2.imshow('frame', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and video writer objects
    cap.release()
    out.release()

    # Close all windows
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(video_path)

    # Define the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    # Variables to store the dimensions of the video frame
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output2.mp4', fourcc, 30,
                          (frame_width, frame_height))

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        # If two faces are detected, assume the person on the right is the second face
        if len(faces) == 2:
            (x1, y1, w1, h1) = faces[0]
            (x2, y2, w2, h2) = faces[1]
            if x1 < x2:
                (x1, y1, w1, h1) = (x2, y2, w2, h2)

            # Draw a rectangle around the person on the right
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

        # Write the output frame to the output video file
        out.write(frame)

        # Display the resulting frame
        # cv2.imshow('frame', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and video writer objects and close all windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # open
    video = VideoFileClip('output2.mp4')
    audio = AudioFileClip(audio2_path)

    # Set the audio to the same duration as the video
    audio = audio.set_duration(video.duration)

    # Mix the audio with the video
    final_clip = video.set_audio(audio)

    # Write the final clip to a new file
    final_clip.write_videofile("finaloutput2.mp4")
    # Create a VideoCapture object to read the video
    cap = cv2.VideoCapture(video_path)
    # close


else:
    cap = cv2.VideoCapture(video_path)

    # Define the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    # Variables to store the dimensions of the video frame
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output1.mp4', fourcc, 30,
                          (frame_width, frame_height))

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        # If two faces are detected, assume the person on the right is the second face
        if len(faces) == 2:
            (x1, y1, w1, h1) = faces[0]
            (x2, y2, w2, h2) = faces[1]
            if x1 > x2:
                (x1, y1, w1, h1) = (x2, y2, w2, h2)

            # Draw a rectangle around the person on the right
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
            # print('hi')

        # Write the output frame to the output video file
        out.write(frame)

        # Display the resulting frame
        # cv2.imshow('frame', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and video writer objects and close all windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    video = VideoFileClip('output1.mp4')
    audio = AudioFileClip(audio1_path)

    # Set the audio to the same duration as the video
    audio = audio.set_duration(video.duration)

    # Mix the audio with the video
    final_clip = video.set_audio(audio)

    # Write the final clip to a new file
    final_clip.write_videofile("finaloutput1.mp4")
    # Create a VideoCapture object to read the video
    cap = cv2.VideoCapture(video_path)

    # Define the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Variables to store the dimensions of the video frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output2.mp4', fourcc, 30,
                          (frame_width, frame_height))

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 2:
            (x1, y1, w1, h1) = faces[0]
            (x2, y2, w2, h2) = faces[1]
            if x1 < x2:
                (x1, y1, w1, h1) = (x2, y2, w2, h2)

            # Draw a rectangle around the person on the right
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
            # print('next hi')

        # Write the output frame to the output video file
        out.write(frame)

        # Display the resulting frame
        # cv2.imshow('frame', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and video writer objects
    cap.release()
    out.release()

    # Close all windows
    cv2.destroyAllWindows()

    video = VideoFileClip('output2.mp4')
    audio = AudioFileClip(audio2_path)

    # Set the audio to the same duration as the video
    audio = audio.set_duration(video.duration)

    # Mix the audio with the video
    final_clip = video.set_audio(audio)

    # Write the final clip to a new file
    final_clip.write_videofile("finaloutput2.mp4")
    # Create a VideoCapture object to read the video
    cap = cv2.VideoCapture(video_path)
