# Plastic Surgery Recognition
### An application that diagnoses whether the nose is aesthetic in real time using tensorflow, keras and dlib.
![XhW](https://i.imgur.com/qHAcfhX.gif)
## Setup This Project
### Install DLIB & OpenCV
- You can install the dlib library by typing ```conda install -c conda-forge dlib``` in your terminal. Anaconda must be installed.
- The easiest way to install opencv is to download it from PyPI. It's going to install the library itself and its prerequisites as well. You can install the opencv library by typing ```pip install opencv-python``` in your terminal.
- Also install tensorflow and keras too
- And then you can run the project with model.py first. It will create the model.h5. Then you can run main.py. Make sure you have **"shape_predictor_68_face_landmarks.dat"** file in your project location.
- [Download shape_predictor_68_face_landmarks.dat](https://github.com/coneypo/Dlib_face_detection_from_camera/raw/master/data/dlib/shape_predictor_68_face_landmarks.dat) I could not add it to this project because it is larger than 25mb
## How this is working?
We created a model with images from our dataset. (resnet50)
Used dlib and dlib's face alignment also opencv.
Opencv gets realtime images from our camera and we diagnose each fragment with our model.
