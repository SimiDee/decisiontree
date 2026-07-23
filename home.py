import streamlit as st 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

st.title('Gender,Music,Prediction')
df = pd.read_csv("music.csv")

st.image('1.jpg') 

with st.expander('music dataset'):
    st.dataframe(df)

x = df.drop(columns=['genre']) 
y = df['genre'] 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2) 

model = DecisionTreeClassifier() 
model.fit(x,y) 


predictions = model.predict(x_test) 

age = st.slider('choose your age',10,70,18)

gender = st.selectbox('Select Gender',['Male','Female']) 
gender_code = 1 if gender == 'Male' else 0 

if st.button('Click to get Classification'):
    prediction = model.predict([[age,gender_code]]) 
    st.success(f'correct: **{prediction[0]}**') 
    
st.video('1.mp4')

