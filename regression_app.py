# import Streamlit for building web app
import streamlit as st
# import pandas for data manipulation
import pandas as pd
#import numpy as np for numerical computing
import numpy as np


# set the title of the Streamlit app
# display a brief description of the app
st.markdown("""<h1 style='color: darkblue;'>Project 1. Regression Models</h1>
            <p>This app builds a machine learning regression model</p><hr>""", unsafe_allow_html=True)

# add subheader
st.subheader("Data preview")
# load the dataset from a CSV file located in the 'data' folder
df = pd.read_csv('data/final.csv')

# display the first five rows of the dataset in the app
st.write('The dataset is loaded. The first five and last five records displayed below:')
st.write(df.head())
st.write(df.tail())

# create variables for rows and columns counts
rows_count = df.shape[0]
columns_count = df.shape[1]
# display dataset shape
st.markdown(f"""
            The dataset contains:
             - **Rows:** { rows_count }
             - **Columns:** { columns_count }
             <hr>
            """, unsafe_allow_html=True)

# add subheader
st.subheader("Linear Regression Model")

# add a short description
st.markdown("""
            Prepare the data for training a **Linear Regression model** 
            by separating the input features ( ***x*** ) from the target variable ( ***y*** ), 
            which is the house price.
            """)

# import the LinearRegression model
from sklearn.linear_model import LinearRegression
