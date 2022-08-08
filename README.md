# HolisticDetection
Holistic detection (tracking face, eye and hand) with an excel output

## Purpose of this script
1. detect the face, eye, posture and hands using mediapipe holisitic
2. capture and save all the static picture with trackings frame-by-frame
3. export an csv file with all the x, y, z coordinate of each region frame-by-frame

## Files
1. face_eye_hand_detection_excel.py --> The main python script
2. Yourvideo.csv --> An example of the export csv file

## output
### static pictures
![Yourvideo](https://user-images.githubusercontent.com/83806848/183409231-fbd4a0fe-7798-4e42-9ee7-19f86f857223.jpg)
1. A screenshot of the static picture with face, eye and hand tracking
2. On the left top corner, it shows the fps of current frame
3. If anything covers the face, the estimate tracking of the face and eye might not be that accurate

### pose_landmarks
1. x and y: Landmark coordinates normalized to [0.0, 1.0] by the image width and height respectively.
2. z: Should be discarded as currently the model is not fully trained to predict depth, but this is something on the roadmap.
3. visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.

## FACE_LANDMARKS
1. A list of 468 face landmarks. Each landmark consists of x, y and z. x and y are normalized to [0.0, 1.0] by the image width and height respectively. 
2. z represents the landmark depth with the depth at center of the head being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.

## LEFT_HAND_LANDMARKS
1. A list of 21 hand landmarks on the left hand. Each landmark consists of x, y and z. x and y are normalized to [0.0, 1.0] by the image width and height respectively. 
2. z represents the landmark depth with the depth at the wrist being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.
