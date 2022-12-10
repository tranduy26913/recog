import streamlit as st
import numpy as np
from PIL import Image
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer
import av

import pickle
from tensorflow import keras
import cv2
import numpy as np
import pickle
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import  load_model

st.set_option('deprecation.showPyplotGlobalUse', False)
import dlib

# Dinh nghia class
class_name = ['00000','10000','20000','50000']

def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    # Dong bang cac layer
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Tao model
    input = Input(shape=(128, 128, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    # Them cac layer FC va Dropout
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation='softmax', name='predictions')(x)

    # Compile
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

# Load weights model da train
my_model = get_model()
my_model.load_weights("weights-49-0.97.hdf5")

# Regco face

st.title("Money Classify")

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    try:
        img = predict(img)
    except:
        pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Dinh nghia class
class_name = ['00000','10000','20000','50000']

def predict(frame):
    image = frame.copy()
    image = cv2.resize(image, dsize=(128, 128))
    image = image.astype('float')*1./255
    # Convert to tensor
    image = np.expand_dims(image, axis=0)

    # Predict
    predict = my_model.predict(image)
    print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
    print(np.max(predict[0],axis=0))
    if (np.max(predict)>=0.8) and (np.argmax(predict[0])!=0):
        # Show image
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (0, 255, 0)
        thickness = 2

        cv2.putText(frame, class_name[np.argmax(predict)], org,font,
                    fontScale, color, thickness, cv2.LINE_AA)
    return frame,class_name[np.argmax(predict[0])],np.max(predict)


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    #print(bytes_data)
    image , predicted,probability = predict(image)
    text = 'Kết quả: ' + str(predicted) +'({probability:.2f}%)'.format(probability=probability*100)
    st.image(image)
    st.subheader(text)
######################################################