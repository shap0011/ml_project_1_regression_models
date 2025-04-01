import streamlit as st
import pandas as pd

st.title('Project 1. Regression Models')

st.write('This app builds a machine learning model')

df = pd.read_csv('data/final.csv')
# df
st.write(df.head())
