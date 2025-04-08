import pandas as pd
from app_module import functions as func
import streamlit as st


# set the title of the Streamlit app
# display a brief description of the app
st.markdown("""<h1 style='color: #94cbe1;'>Project 1. Regression Models</h1>
            <p>This app builds a machine learning regression model</p><hr>""", unsafe_allow_html=True)

# st.write(sklearn.__version__)

# add subheader
st.subheader("Data preview")
# load the dataset from a CSV file located in the 'data' folder
df = func.load_data('data/final.csv')

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
# model = LinearRegression()
# # fit the model using the training data (input features and target)
# lrmodel = model.fit(x_train, y_train)
# Train model
lrmodel = func.train_linear_regression(x_train, y_train)

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

# display subheader
st.subheader("Decision Tree Model")

# import decision tree model
from sklearn.tree import DecisionTreeRegressor

# # create an instance of the class
# dt = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=567)
# # train the model
# dtmodel = dt.fit(x_train,y_train)
# # Train model
# lrmodel = func.train_linear_regression(x_train, y_train)

# st.write("Make prediction using the train set and evaluate the model:")

# # make predictions using the training set
# ytrain_pred = lrmodel.predict(x_train)


# # evaluate the model
# train_mae = mean_absolute_error(ytrain_pred, y_train)
# st.write(f"MAE (x_train set): `{train_mae}`")

# st.write("Make prediction using the test set and evaluate the model:")
# # make predictions using the test set
# ytest_pred = dtmodel.predict(x_test)

# # evaluate the model
# test_mae = mean_absolute_error(ytest_pred, y_test)
# st.write(f"MAE (x_test set): `{test_mae}`")

# Train models
lrmodel = func.train_linear_regression(x_train, y_train)
dtmodel = func.train_decision_tree(x_train, y_train)

# Predict and evaluate with lrmodel
st.write("Linear Regression Model:")
# ytrain_pred = lrmodel.predict(x_train)
# train_mae = mean_absolute_error(ytrain_pred, y_train)
# st.write(f"MAE (x_train set): `{train_mae}`")
# Evaluate on train set
train_mae = func.evaluate_model(lrmodel, x_train, y_train, dataset_name="Training Set")

# ytest_pred = lrmodel.predict(x_test)
# test_mae = mean_absolute_error(ytest_pred, y_test)
# st.write(f"MAE (x_test set): `{test_mae}`")
# Evaluate on test set
test_mae = func.evaluate_model(lrmodel, x_test, y_test, dataset_name="Test Set")

# Predict and evaluate with dtmodel
st.write("Decision Tree Model:")
# ytrain_pred_dt = dtmodel.predict(x_train)
# train_mae_dt = mean_absolute_error(ytrain_pred_dt, y_train)
# st.write(f"Decision Tree MAE (x_train set): `{train_mae_dt}`")
train_mae_dt = func.evaluate_model(dtmodel, x_train, y_train, dataset_name="Training Set (Decision Tree)")

# ytest_pred_dt = dtmodel.predict(x_test)
# test_mae_dt = mean_absolute_error(ytest_pred_dt, y_test)
# st.write(f"Decision Tree MAE (x_test set): `{test_mae_dt}`")
test_mae_dt = func.evaluate_model(dtmodel, x_test, y_test, dataset_name="Test Set (Decision Tree)")


# display subheader
st.subheader("How do I know if my model is Overfitting or Generalized?")

# make predictions on train set
ytrain_pred = dtmodel.predict(x_train)

# import mean absolute error metric
from sklearn.metrics import mean_absolute_error

# evaluate the model
train_mae = mean_absolute_error(ytrain_pred, y_train)
st.write(f"MAE (x_train set): `{train_mae}`")

# display subheader
st.subheader("Plot the tree")
st.write("Get the features")

# get the features
features = dtmodel.feature_names_in_
# st.write(", ".join(map(str, features)))
st.markdown("<br>".join(map(str, features)), unsafe_allow_html=True)

# display subheader
st.subheader("Plot the tree")

# plot the tree
import matplotlib.pyplot as plt
from sklearn import tree

# Show the plot in Streamlit
st.write("Decision Tree Visualization")

# # Create a figure and axis
# fig, ax = plt.subplots(figsize=(30, 10))  # Optional: control figure size

# # Plot the tree
# tree.plot_tree(dtmodel, feature_names=dtmodel.feature_names_in_, filled=True, rounded=True, fontsize=10, ax=ax)

# st.pyplot(fig)

# # Save the plot to a file
# plt.savefig('tree.png', dpi=300)

# Plot decision tree
fig = func.plot_tree_model(dtmodel, dtmodel.feature_names_in_)

# display subheader
st.subheader("Random Forest Model")

# import decision tree model
from sklearn.ensemble import RandomForestRegressor

# create an instance of the model
rf = RandomForestRegressor(n_estimators=200, criterion='absolute_error')

# train the model
rfmodel = rf.fit(x_train,y_train)

# make prediction on train set
ytrain_pred = rfmodel.predict(x_train)

# make predictions on the x_test values
ytest_pred = rfmodel.predict(x_test)

# evaluate the model
test_mae = mean_absolute_error(ytest_pred, y_test)
st.write(f"MAE (x_test set): `{test_mae}`")

 # display subheader
st.subheader("Pickle:")
st.markdown("""
- The pickle module implements a powerful algorithm for serializing and de-serializing a Python object structure.
- The saving of data is called Serialization, and loading the data is called De-serialization.

**Pickle module provides the following functions:**

- **pickle.dump** to serialize an object hierarchy, you simply use `dump()`.
- **pickle.load** to deserialize a data stream, you call the `load()` function.
""")

# import pickle to save model
import pickle

# Save the trained model on the drive
pickle.dump(dtmodel, open('RE_Model','wb'))

# Load the pickled model
RE_Model = pickle.load(open('RE_Model','rb'))

# Use the loaded pickled model to make predictions
RE_Model.predict([[2012, 216, 74, 1, 1, 618, 2000, 600, 1, 0, 0, 6, 0, 1]])

st.write("Use the loaded pickled model to make predictions")
x_test_head_1 = x_test.head(1)
st.dataframe(x_test_head_1)
