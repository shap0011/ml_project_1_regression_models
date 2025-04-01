# import Streamlit for building web app
import streamlit as st
# import pandas for data manipulation
import pandas as pd
#import numpy as np for numerical computing
import numpy as np


# set the title of the Streamlit app
st.title('Project 1. Regression Models')

# display a brief description of the app
st.write('This app builds a machine learning model')

# load the dataset from a CSV file located in the 'data' folder
df = pd.read_csv('data/final.csv')

# display the first five rows of the dataset in the app
st.write('The dataset is loaded. The first five and last five records displayed below:')
st.write(df.head())
st.write(df.tail())


rows_count = df.shape[0]
columns_count = df.shape[1]
st.write(f'The dataset contains: \nRows: {rows_count}\nColumns: {columns_count}')
