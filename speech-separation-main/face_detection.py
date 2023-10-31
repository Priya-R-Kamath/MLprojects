import cv2
import moviepy.editor as mp

# Load the Haar Cascade face classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


video_path = input("Give the path to the video")
# Open a video capture object
cap = cv2.VideoCapture(video_path)

# Create a VideoFileClip object using the video file
clip = mp.VideoFileClip(video_path)


audio_path = input("Give the path to the audio")
# Create an AudioClip object using the audio file
audio = mp.AudioFileClip(audio_path)

# Loop through each frame of the video
frames = []
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Stop the loop if we have reached the end of the video
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    # Loop through each face and draw a rectangle around it
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Add the frame to the list of frames
    frames.append(frame)

# Create a new video clip from the list of frames
video = mp.ImageSequenceClip(frames, fps=clip.fps)

# Set the audio of the new video clip to the separate audio file
video = video.set_audio(audio)

# Write the new video clip to a file
video.write_videofile('new_video.mp4')
