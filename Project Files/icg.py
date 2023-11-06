import pickle
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Activation, Dropout, Dense, Input, Layer
from tensorflow.keras.layers import Embedding, LSTM, add, Reshape, concatenate
from tensorflow.keras.models import Model, load_model

#Required Functions

def feature_extract(path):
    img = load_img(path,target_size=(224,224))
    img = img_to_array(img)
    img = img/225.
    img = np.expand_dims(img,axis=0)
    return fe.predict(img,verbose=0)

def idx_to_word(integer,tokenizer):
    
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

def predict_caption(model, tokenizer, max_length, feature):
    
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = model.predict([feature,sequence],verbose = 0)
        y_pred = np.argmax(y_pred)
        
        word = idx_to_word(y_pred, tokenizer)
        
        if word is None:
            break
            
        in_text+= " " + word
        
        if word == 'endseq':
            break
            
    return in_text 

#Initialization

#Image feature extraction model
model = DenseNet201()
fe = Model(inputs=model.input, outputs=model.layers[-2].output)

#Load Tokenizer
tokenizer = Tokenizer()
with open('./ModelData/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#Other Parameters
max_length = 36 #or max(len(caption.split()) for caption in captions)
vocab_size = len(tokenizer.word_index) + 1

#Prediction Model Initilization

input1 = Input(shape=(1920,))
input2 = Input(shape=(max_length,))

img_features = Dense(256, activation='relu')(input1)
img_features_reshaped = Reshape((1, 256), input_shape=(256,))(img_features)

sentence_features = Embedding(vocab_size, 256, mask_zero=False)(input2)
merged = concatenate([img_features_reshaped,sentence_features],axis=1)
sentence_features = LSTM(256)(merged)
x = Dropout(0.5)(sentence_features)
x = add([x, img_features])
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(vocab_size, activation='softmax')(x)

caption_model = Model(inputs=[input1,input2], outputs=output)
caption_model.compile(loss='categorical_crossentropy',optimizer='adam')

#Load Weights
caption_model = load_model('./ModelData/cmodel.h5')