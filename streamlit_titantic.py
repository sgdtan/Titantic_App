# ============
# Import Lib
# ============
import pandas as pd
import numpy as np
import streamlit as st

import cv2
from PIL import Image
import os
import json
#from numpyencoder import NumpyEncoder
from catboost import Pool, CatBoostClassifier
path = os.getcwd() + '/titantic_catmodel'
import time

print(path)

cat = CatBoostClassifier()
cat.load_model(path)



# =======
# title
# =======
st.title('''Survive the Titantic?''')


# ==========
# subheader
# ==========
st.subheader('This app calculates the probability of surviving Titantic for a given passenger profile')
st.write('      ')

# ==========================================
# Detailed explaination
# ==========================================
st.write('''
Enter Passenger Details here:

''')


# create line break

st.write(' ')

G = st.selectbox('Gender', ('Male', 'Female') )

if G == 'Male':
    Gender = 1
else:
    Gender = 0

st.write(' ')

Age = int (st.slider('Age',1,130,25) )

st.write(' ')

TC = st.selectbox('Ticket Class', ('First Class', 'Second Class', 'Third Class') )

if TC == 'First Class':
    TicketClass = 1
elif TC == 'Second Class':
    TicketClass = 2
elif TC == 'Third Class':
    TicketClass = 3


st.write(' ')

Fare = st.slider('Ticket Fare', 1, 550)

input = np.array([TicketClass, Gender, Age, Fare])

# Button to run Model
if st.button('Run'):
    with st.spinner('Running...'):
        #time.sleep(1)
        survival_prob = int(cat.predict_proba(input)[1] * 100)

    st.write('This Passenger has a Titantic Survival probability of', survival_prob, ' percent.')










else:
    st.write('Run ML Model to get survivial probability')


