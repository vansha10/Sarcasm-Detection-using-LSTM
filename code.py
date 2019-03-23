# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 19:34:29 2019

@author: Vansh
"""

import pandas as pd
pd.set_option('display.max_colwidth', -1)
sar_acc = pd.read_json('Sarcasm_Headlines_Dataset.json',lines=True)
import re
sar_acc['source'] = sar_acc['article_link'].apply(lambda x: re.findall(r'\w+', x)[2])
sar_acc.head()

#Getting X and Y ready
from sklearn.preprocessing import LabelEncoder
X = sar_acc.headline
Y = sar_acc.is_sarcastic
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)

# Split into Training and Test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

# Processing the data for the model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

# Defining the RNN structure for the model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.models import Model
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

# Compiling the model
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

# Fitting the model on training data
from keras.callbacks import EarlyStopping
model.fit(sequences_matrix,Y_train,batch_size=100,epochs=5,
          validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

# Testing the model
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

# Model Accuracy
accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

to_pred = input("Enter the string to predict: ")
test = tok.texts_to_sequences([to_pred])
test_matrix = sequence.pad_sequences(test,maxlen=max_len)

res = model.predict(test_matrix)

print("There is a %.2f%% chance that this sentence was sarcastic." %(res[0][0]*100))


#Speech to text
import speech_recognition as sr

r = sr.Recognizer()

mic = sr.Microphone(device_index=0)

with mic as source:
    r.adjust_for_ambient_noise(source)
    print("Listening...")
    audio = r.listen(source)
    
try:
    print("Recognizing...")
    to_pred_audio = r.recognize_google(audio)
    print("You said:\n" + to_pred_audio)
    
    test = tok.texts_to_sequences([to_pred_audio])
    test_matrix = sequence.pad_sequences(test,maxlen=max_len)

    res = model.predict(test_matrix)

    print("There is a %.2f%% chance that this sentence was sarcastic." %(res[0][0]*100))
except:
    print("Not recognized!")

with open("microphone-results.wav", "wb") as f:
        f.write(audio.get_wav_data())

 

