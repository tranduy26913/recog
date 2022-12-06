import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import numpy as np
import joblib
import cv2
from streamlit_webrtc import webrtc_streamer
import av

import pickle
from tensorflow import keras
import time
import mtcnn
st.set_option('deprecation.showPyplotGlobalUse', False)
import dlib
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
#detector = mtcnn.MTCNN()
# detect faces in the image
#faces = detector.detect_faces(pixels)
facenet_model = keras.models.load_model('models/facenet_keras.h5')
dest_size = (160, 160)
print(dest_size)
# Load SVM model từ file
pkl_filename = 'faces_svm.pkl'
with open(pkl_filename, 'rb') as file:
    svm_model = pickle.load(file)

# Load ouput_enc từ file để hiển thị nhãn
pkl_filename = 'output_enc.pkl'
with open(pkl_filename, 'rb') as file:
    output_enc = pickle.load(file)

# Regco face


st.title("My first Streamlit app")


def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    try:
        img = predict(img)
    except:
        pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")


def get_embedding(img):
    # scale pixel values
    img = img.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = img.mean(), img.std()
    img = (img - mean) / std

    # transform face into one sample
    samples = np.expand_dims(img, axis=0)

    # make prediction to get embedding
    return facenet_model.predict(samples)[0]


def predict(frame):
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 0)
    #print(frame.shape)

    #dets = detector.detect_faces(frame)
    print(len(dets))
    if len(dets) > 0:
        rect = dets[0]
        x = rect.left()
        y = rect.top()
        w = rect.right()
        h = rect.bottom()
        # x, y, w, h = rect['box']
        # w = x + w
        # h = y + h

        face = frame[y:h, x:w]
        print(face.shape)
        face = cv2.resize(face, dsize=dest_size)
        # Lây face embeding
        face_emb = get_embedding(np.array(face))
        # Chuyển thành tensor
        face_emb = np.expand_dims(face_emb, axis=0)
        # Predict qua SVM
        predict = svm_model.predict_proba(face_emb)
        # Tính xác suất chính xác khi dự đoán
        probability = round(np.max(predict) * 100, 2)
        # Lấy label
        label = [np.argmax(predict)]
        print(label)
        # Lấy nhãn và viết lên ảnh
        predict_names = output_enc.inverse_transform(label)

        if predict_names != None:
            if probability > 0:  # chỉ những dự đoán có xác suất trên 70% mới giữ lại
                text = predict_names[0]+f'({probability}%)'
            else:
                text = "Khong xac dinh"
            frame = cv2.rectangle(
                frame, (x - 20, y - 50), (w+20, h+20), (36, 255, 12), 2)
            cv2.putText(
                frame, text, (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame


# webrtc_streamer(key="example",
#                 video_frame_callback=callback,
#                 rtc_configuration={  # Add this line
#                     "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#                 })
webrtc_streamer(key="example",
                video_frame_callback=callback,
               )
