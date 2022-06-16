# This is a sample Python script.
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
import pickle
import PIL
from PIL import Image
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.utils import load_img
from keras.utils import img_to_array
from keras.utils import array_to_img
from keras.utils import pad_sequences
import json
import requests
import time
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

vgg_model = VGG16()
cust_img_model = keras.Model(inputs= vgg_model.input,outputs = vgg_model.layers[-2].output)

def get_reduced_features():

    img = load_img('input.jpg', target_size=(224, 224))
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    features = cust_img_model.predict(img_arr)
    print("features")
    print("---------")
    print(features)
    return features

def generate_caption_using_api( photo, max_caption_length, tokenizer):
    url = 'http://54.161.181.223/v1/models/tensorflow_models:predict'
    text = 'startseq'
    p = [float(x) for x in photo[0]]
    url = 'http://54.161.181.223/v1/models/tensorflow_models:predict'
    # photo = train_features['2258662398_2797d0eca8']
    # sendtoapi(photo,text)
    for i in range(max_caption_length):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=max_caption_length)
        s = [float(x) for x in seq[0]]
        img_tensor_api = {'input_2': p, 'input_3': s}
        json_data = {
            "instances": [img_tensor_api]
        }
        response = requests.post(url, json=json_data)
        time.sleep(1)
        y_pred = response.json()['predictions'][0]
        y_pred = np.argmax(y_pred)
        predicted_word = tokenizer.sequences_to_texts([[y_pred]])[0]
        # print('predicted word:',predicted_word)
        if predicted_word is None:
            break
        text = text + ' ' + predicted_word

        if predicted_word == 'endseq':
            break
    # print(text)
    tmp = text.split()[1:-1]
    tmp[0] = tmp[0].capitalize()
    # print(tmp)
    captions = " ".join(tmp)
    return captions

def runModel(uploaded_file):

    input = Image.open(uploaded_file)
    input.save('input.jpg')
    features = get_reduced_features()
    max_caption_length = 34
    captions = generate_caption_using_api(features,max_caption_length,tokenizer)
    print(captions)
    st.subheader("And here is what your picture is all about:")
    st.write(captions)
    print("----Sending to api-----")
    print(captions)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


st.title('Caption Generator')
st.header("Pick an image and my CaptionGen Model describes the image!!!Pure Magic:P")



uploaded_file = st.file_uploader("CHOOSE YOUR FILE/IMAGE:")
# if(uploaded_file):



if uploaded_file:
    st.subheader("You chose this picture!")
    st.image(uploaded_file)
    runModel(uploaded_file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
