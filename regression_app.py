# import Streamlit for building web app
import streamlit as st
# import pandas for data manipulation
import pandas as pd
#import numpy as np for numerical computing
import numpy as np


# set the title of the Streamlit app
# display a brief description of the app
st.markdown("""<h1 style='color: #94cbe1;'>Project 1. Regression Models</h1>
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
            by separating the input features ( `x` ) from the target variable ( `y` ), 
            which is the house price.
            """)

# import the LinearRegression model
from sklearn.linear_model import LinearRegression

# separate the input features (all columns except 'price')
x = df.drop('price', axis=1)

# store the target variable (price) in y
y = df['price']


# display the first few rows of the input features
input_features_top_5 = x.head()
# display subheader text
st.markdown("###### First five row of input features")
# display dataframe table
st.dataframe(input_features_top_5)



# display the first few values of the target variable
target_variable_top_5 = y.head()
# display subheader text
st.markdown("###### First few values of the target variable")
# display dataframe table
st.dataframe(target_variable_top_5)

# display model training instructions and explain the purpose of train-test split
st.markdown("""
### Train-Test Split

- **Training set**: Used to fit and tune the model  
- **Test set**: Held back to evaluate model performance on unseen data  
- `train_test_split()` helps split the dataset into these randomized subsets
""")

# import the train_test_split function to split the dataset into training and test sets
from sklearn.model_selection import train_test_split

# create two columns side by side in the Streamlit layout
col1, col2 = st.columns(2)

# first attempt: split the data randomly (20% test, 80% train)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

# count and display how many 'Bungalow' type properties are in the training set
# this helps check if the class distribution is balanced
x_train_bungalow_random = x_train.property_type_Bunglow.value_counts()
# populate the first column with the distribution before stratified split
with col1:
    st.markdown("##### Before Stratified Split")  # add a subheader
    st.dataframe(x_train_bungalow_random)        # display the random split distribution

# second attempt: split the data while preserving the distribution of 'Bungalow' property type
# stratify ensures proportional representation in both training and test sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=x.property_type_Bunglow)

# count and display 'Bungalow' values again after stratified split to compare distribution
x_train_bungalow_stratified = x_train.property_type_Bunglow.value_counts()
# populate the second column with the distribution after stratified split
with col2:
    st.markdown("##### After Stratified Split")   # add a subheader
    st.dataframe(x_train_bungalow_stratified)    # display the stratified split distribution
    
