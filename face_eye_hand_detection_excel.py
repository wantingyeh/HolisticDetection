import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import uuid
import os

#### purpose: detect the face, eye, posture and hands using mediapipe holisitic
### getting the values of eye region, face region and hand region
### Automatically capture the static pictures with skeleton frame-by-frame
### Produce an excel output for x, y, z coordinate for each region frame-by-frame

# set path and output files

VIDEO_PATH = "Yourvideo.mp4"
path_image = "Yourvideo"
outputfile_csv = "Yourvideo_output.csv"


## create a folder for saving pics
os.mkdir(path_image)


# prep for annotation
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose_face = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
              'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
              'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
 
pose_hand = ['WRIST', 'THUMB_CPC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP',
               'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
               'RING_FINGER_MCP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']
 
pose_hand_2 = ['WRIST2', 'THUMB_CPC2', 'THUMB_MCP2', 'THUMB_IP2', 'THUMB_TIP2', 'INDEX_FINGER_MCP2', 'INDEX_FINGER_PIP2', 'INDEX_FINGER_DIP2', 'INDEX_FINGER_TIP2', 'MIDDLE_FINGER_MCP2',
               'MIDDLE_FINGER_PIP2', 'MIDDLE_FINGER_DIP2', 'MIDDLE_FINGER_TIP2', 'RING_FINGER_PIP2', 'RING_FINGER_DIP2', 'RING_FINGER_TIP2',
               'RING_FINGER_MCP2', 'PINKY_MCP2', 'PINKY_PIP2', 'PINKY_DIP2', 'PINKY_TIP2']

# create a dataset to store the dataframe
alldata = []

# create counter for getting fps
counter = 0

cap = cv2.VideoCapture(VIDEO_PATH)
# holistic detection
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  
  while cap.isOpened():
      counter += 1
      #print(f'processing frame: {counter}')
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
        break
 
    # To improve performance, optionally mark the image as not writeable to pass by reference
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = holistic.process(image) 
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the detection annotations on the image.
      mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
            
      mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())

      mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())
    
      mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())

    #  storing the values from the results
      data_holisitc = {}
      if results.face_landmarks: 
        for i in range(len(pose_face)):
            results.face_landmarks.landmark[i].x = results.face_landmarks.landmark[i].x * image.shape[0]
            results.face_landmarks.landmark[i].y = results.face_landmarks.landmark[i].y * image.shape[1]
            data_holisitc.update(
                {pose_face[i] : results.face_landmarks.landmark[i]}
            )
        #alldata.append(data_holisitc)

      if results.right_hand_landmarks:
        for i in range(len(pose_hand)):
            results.right_hand_landmarks.landmark[i].x = results.right_hand_landmarks.landmark[i].x * image.shape[0]
            results.right_hand_landmarks.landmark[i].y = results.right_hand_landmarks.landmark[i].y * image.shape[1]
            data_holisitc.update(
                {pose_hand[i] : results.right_hand_landmarks.landmark[i]}
            )
        #alldata.append(data_holisitc)

      if results.left_hand_landmarks:
        for i in range(len(pose_hand_2)):
            results.left_hand_landmarks.landmark[i].x = results.left_hand_landmarks.landmark[i].x * image.shape[0]
            results.left_hand_landmarks.landmark[i].y = results.left_hand_landmarks.landmark[i].y * image.shape[1]
            data_holisitc.update(
                {pose_hand_2[i] : results.left_hand_landmarks.landmark[i]}
            )
    
      alldata.append(data_holisitc)
      #print(alldata)

    
    # Displaying FPS on the image
      cv2.putText(image, str(int(counter))+" FPS", (10, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (13, 0, 255), 1)

    # video with trackers on
      cv2.imshow('MediaPipe Face and hand Detection', image)
      #print(data_holisitc)

    # Save images and data   
      cv2.imwrite(path_image + "\\frame%d.jpg" % counter, image)
    
      if cv2.waitKey(10) & 0xFF == ord('q'):  # press q key to stop the video   
        break

  # save data in csv
  df = pd.DataFrame(alldata)
  df.to_csv(outputfile_csv, index = True)

#print(f'Frames: {counter}')
#print(df.head())  
cap.release()


## output
# pose_landmarks
# x and y: Landmark coordinates normalized to [0.0, 1.0] by the image width and height respectively.
# z: Should be discarded as currently the model is not fully trained to predict depth, but this is something on the roadmap.
# visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.

# FACE_LANDMARKS
# A list of 468 face landmarks. Each landmark consists of x, y and z. x and y are normalized to [0.0, 1.0] by the image width and height respectively. 
# z represents the landmark depth with the depth at center of the head being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.

#LEFT_HAND_LANDMARKS
#A list of 21 hand landmarks on the left hand. Each landmark consists of x, y and z. x and y are normalized to [0.0, 1.0] by the image width and height respectively. 
# z represents the landmark depth with the depth at the wrist being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.
