# import Streamlit for building web app
import streamlit as st
# import pandas for data manipulation
import pandas as pd
#import numpy as np for numerical computing
import numpy as np
import sklearn



# set the title of the Streamlit app
# display a brief description of the app
st.markdown("""<h1 style='color: #94cbe1;'>Project 1. Regression Models</h1>
            <p>This app builds a machine learning regression model</p><hr>""", unsafe_allow_html=True)

# st.write(sklearn.__version__)

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

# display the section header    
st.markdown("#### Preview of Training Features and Target Variable")

# display the first five rows of the training input features (x_train)
# get top 5 rows from training feature set
x_train_head = x_train.head()
# add a table title
st.markdown("##### First 5 Rows of Training Features (x_train)")
# show table
st.dataframe(x_train_head)

# display the first five rows of the training target variable (y_train)
# get top 5 values from training target variable
y_train_head = y_train.head()
# add a table title
st.markdown("##### First 5 Values of Training Target (y_train)")
# show table
st.dataframe(y_train_head)

# display the shape (rows, columns) of training and test datasets
# display the section title
st.markdown("##### Dataset Shapes: Training and Test Sets")

# get the shape (row count, column count) of each dataset
# input features for training
x_train_shape = x_train.shape
# target variable for training
y_train_shape = y_train.shape
# input features for testing
x_test_shape = x_test.shape
# target variable for testing
y_test_shape = y_test.shape

st.markdown(f"Dimensions of input features for training `x_train`: {x_train_shape}")
st.markdown(f"Dimensions of target variable for training `y_train`: {y_train_shape}")
st.markdown(f"Dimensions of input features for testing `x_test`: {x_test_shape}")
st.markdown(f"Dimensions of target variable for testing `y_test`: {y_test_shape}")

# training the Linear Regression Model
# display the section title
st.markdown("##### Train the Linear Regression Model")

# create an instance of the LinearRegression model
model = LinearRegression()
# fit the model using the training data (input features and target)
lrmodel = model.fit(x_train, y_train)

# Access the learned coefficients (weights) of the trained model
lrmodel_coef_ = lrmodel.coef_

# display the learned coefficients
st.markdown(f"""
            Learned coefficients (weights) of the trained model:
            <br>
            `{ lrmodel_coef_ }`""", unsafe_allow_html=True)

# Access the model's intercept (bias term)
lrmodel_intercept_ = lrmodel.intercept_

# display the model's intercept
st.markdown(f"""
            Model's intercept:
            <br>
            `{ lrmodel_intercept_ }`
            """, unsafe_allow_html=True)

# preview the first row of the training features which is used for demonstration/prediction
x_train_head_first = x_train.head(1)

# display the first row of input features in the app
st.markdown("###### First Row of Training Features (x_train)")
st.dataframe(x_train_head_first)

# making predictions and evaluating model performance
# display the section title
st.markdown("### Make Predictions and Evaluate Model Performance")

# predict house prices on the training set using the trained model
train_pred = lrmodel.predict(x_train)

# display predicted vs actual values
# display predicted values
st.markdown(f"**Predicted values (first 10):** `{train_pred[:10].tolist()}`")
# display actual values
st.markdown(f"**Actual values (first 10):** `{y_train.head(10).tolist()}`")



# import the evaluation metric: Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error

# calculate MAE between predicted and actual house prices
train_mae = mean_absolute_error(train_pred, y_train)

# print the training error to the console (for debugging/logging)
st.markdown(f"Train error is: `{train_mae}`")

# display the model's coefficients
st.markdown(f"""
            Learned coefficients (weights) of the trained model:
            <br>
            `{ lrmodel_coef_ }`""", unsafe_allow_html=True)


# Our model is still not good because we need a model with Mean Absolute Error < $70,000
# Note - We have not scaled the features and not tuned the model.

# model interpretation
# display the section title
st.markdown("#### Model Interpretation")
st.markdown("""
            <ul>
                <li>The built model's performance is not ideal yet</li>
                <li>Goal: MAE should be below $70,000</li>
                <li>The model features have not yet scaled and the model not tuned</li>
            </ul>
            """, unsafe_allow_html=True)

# display subheader
st.subheader("How Each Feature Affects Price")

# get column names (input features)
column_names = x_train.columns

# get learned coefficients from the model
lrmodel_coef = lrmodel.coef_

# create a DataFrame matching each feature name with its coefficient
coefficients_df = pd.DataFrame({
    'Feature': column_names,
    'Coefficient': lrmodel_coef
})

# Display the result in Streamlit
st.markdown("##### Feature Coefficients")
st.dataframe(coefficients_df)
