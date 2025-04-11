import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'info'

import logging

# Configure logging manually again
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

import pandas as pd
from app_module import functions as func
import streamlit as st

try:  
    
    #-------- page setting, header, intro ---------------------
      
    # Set page configuration
    st.set_page_config(page_title="🏡 House Price Prediction App", layout="wide")
    # set the title of the Streamlit app
    # display a brief description of the app
    # st.markdown("""<h1 style='color: #94cbe1;'>Project 1. Regression Models</h1>
    #             <p>This app builds a machine learning regression model</p>""", unsafe_allow_html=True)
    
    st.markdown("<h1 style='color: #3498db;'>🏡 House Price Prediction App</h1>", unsafe_allow_html=True)
    
    #-------- the app overview -----------------------------
    
    st.markdown("""
    ### Overview
    Welcome to the **House Price Prediction App**!

    This web application allows users to:
    - Explore a real estate dataset 📄
    - Understand the relationship between house features and prices 🏠
    - Train and evaluate different machine learning models (Linear Regression, Decision Tree, Random Forest) 🤖
    - **Predict** house prices based on customizable property features 🎯
    """)
    
    #-------- user instructions -------------------------------
    
    st.markdown("""
    ### How to Use This App

    1. **Review the Overview**: Understand the dataset and models used.
    2. **Input Property Details**: 
    - Fill in the property characteristics (like Year Built, Lot Size, Living Area, etc.) in the form below.
    3. **Submit the Form**:
    - Click on the **"Predict House Price"** button.
    4. **View Results**:
    - See the **predicted house price** instantly on the screen.

    > ⚡ *Tip: You can adjust the property features and re-submit to see how the price changes!*
    """)

    #-------- the dataset loading -----------------------------

    # add subheader
    # st.subheader("Data preview")
    # load the dataset from a CSV file located in the 'data' folder
    try:
        df = func.load_data('data/final.csv')
        logging.info("Dataset loaded successfully!")
        logging.warning("[INFO] Dataset loaded successfully.") # for the Streamlit web app
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        st.error("Dataset file not found. Please check the 'data/final.csv' path.")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        st.error("An unexpected error occurred while loading the dataset.")


    # display the first five rows of the dataset in the app
    # st.write('The dataset is loaded.')
    # st.write('The first five and last five records displayed below:')
    # st.write(df.head())
    # st.write(df.tail())
    
    #-------- the linear regression model ---------------------
    
    # create variables for rows and columns counts
    rows_count = df.shape[0]
    columns_count = df.shape[1]
    # display dataset shape
    # st.markdown(f"""
    #             The dataset contains:
    #             - **Rows:** { rows_count }
    #             - **Columns:** { columns_count }
    #             <hr>
    #             """, unsafe_allow_html=True)

    # add subheader
    # st.subheader("Linear Regression Model")

    # add a short description
    # st.markdown("""
    #             Prepare the data for training a **Linear Regression model** 
    #             by separating the input features ( `x` ) from the target variable ( `y` ), 
    #             which is the house price.
    #             """)

    # import the LinearRegression model
    from sklearn.linear_model import LinearRegression

    # separate the input features (all columns except 'price')
    x = df.drop('price', axis=1)

    # store the target variable (price) in y
    y = df['price']

    # display the first few rows of the input features
    input_features_top_5 = x.head()
    # display subheader text
    # st.markdown("###### First five row of input features")
    # # display dataframe table
    # st.dataframe(input_features_top_5)

    # display the first few values of the target variable
    target_variable_top_5 = y.head()
    # display subheader text
    # st.markdown("###### First few values of the target variable")
    # # display dataframe table
    # st.dataframe(target_variable_top_5)
    
    #-------- train-test split ---------------------

    # display model training instructions and explain the purpose of train-test split
    # st.markdown("""
    # ### Train-Test Split

    # - **Training set**: Used to fit and tune the model  
    # - **Test set**: Held back to evaluate model performance on unseen data  
    # - `train_test_split()` helps split the dataset into these randomized subsets
    # """)

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
    # with col1:
    #     st.markdown("##### Before Stratified Split")  # add a subheader
    #     st.dataframe(x_train_bungalow_random)        # display the random split distribution

    # second attempt: split the data while preserving the distribution of 'Bungalow' property type
    # stratify ensures proportional representation in both training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=x.property_type_Bunglow)

    # count and display 'Bungalow' values again after stratified split to compare distribution
    x_train_bungalow_stratified = x_train.property_type_Bunglow.value_counts()
    # populate the second column with the distribution after stratified split
    # with col2:
    #     st.markdown("##### After Stratified Split")   # add a subheader
    #     st.dataframe(x_train_bungalow_stratified)    # display the stratified split distribution

    # display the section header    
    # st.markdown("#### Preview of Training Features and Target Variable")

    # display the first five rows of the training input features (x_train)
    # get top 5 rows from training feature set
    x_train_head = x_train.head()
    # add a table title
    # st.markdown("##### First 5 Rows of Training Features (x_train)")
    # # show table
    # st.dataframe(x_train_head)

    # display the first five rows of the training target variable (y_train)
    # get top 5 values from training target variable
    y_train_head = y_train.head()
    # add a table title
    # st.markdown("##### First 5 Values of Training Target (y_train)")
    # # show table
    # st.dataframe(y_train_head)

    # display the shape (rows, columns) of training and test datasets
    # display the section title
    # st.markdown("##### Dataset Shapes: Training and Test Sets")

    # get the shape (row count, column count) of each dataset
    # input features for training
    x_train_shape = x_train.shape
    # target variable for training
    y_train_shape = y_train.shape
    # input features for testing
    x_test_shape = x_test.shape
    # target variable for testing
    y_test_shape = y_test.shape

    # st.markdown(f"Dimensions of input features for training `x_train`: {x_train_shape}")
    # st.markdown(f"Dimensions of target variable for training `y_train`: {y_train_shape}")
    # st.markdown(f"Dimensions of input features for testing `x_test`: {x_test_shape}")
    # st.markdown(f"Dimensions of target variable for testing `y_test`: {y_test_shape}")

    # training the Linear Regression Model
    # display the section title
    # st.markdown("##### Train the Linear Regression Model")

    # Train model
    # lrmodel = func.train_linear_regression(x_train, y_train)
    try:
        lrmodel = func.train_linear_regression(x_train, y_train)
        logging.info("Linear Regression model trained successfully.")
    except Exception as e:
        logging.error(f"Failed to train Linear Regression model: {e}")
        st.error("Error training Linear Regression model.")


    # Access the learned coefficients (weights) of the trained model
    lrmodel_coef_ = lrmodel.coef_

    # display the learned coefficients
    # st.markdown(f"""
    #             Learned coefficients (weights) of the trained model:
    #             <br>
    #             `{ lrmodel_coef_ }`""", unsafe_allow_html=True)

    # Access the model's intercept (bias term)
    lrmodel_intercept_ = lrmodel.intercept_

    # display the model's intercept
    # st.markdown(f"""
    #             Model's intercept:
    #             <br>
    #             `{ lrmodel_intercept_ }`
    #             """, unsafe_allow_html=True)

    # preview the first row of the training features which is used for demonstration/prediction
    x_train_head_first = x_train.head(1)

    # display the first row of input features in the app
    # st.markdown("###### First Row of Training Features (x_train)")
    # st.dataframe(x_train_head_first)
    
    #-------- Make Predictions and Evaluate Model Performance ---------------------

    # making predictions and evaluating model performance
    # display the section title
    # st.markdown("### Make Predictions and Evaluate Model Performance")

    # predict house prices on the training set using the trained model
    train_pred = lrmodel.predict(x_train)

    # display predicted vs actual values
    # display predicted values
    # st.markdown(f"**Predicted values (first 10):** `{train_pred[:10].tolist()}`")
    # display actual values
    # st.markdown(f"**Actual values (first 10):** `{y_train.head(10).tolist()}`")

    # import the evaluation metric: Mean Absolute Error (MAE)
    from sklearn.metrics import mean_absolute_error

    # calculate MAE between predicted and actual house prices
    train_mae = mean_absolute_error(train_pred, y_train)

    # print the training error to the console (for debugging/logging)
    # st.markdown(f"Train error is: `{train_mae}`")

    # display the model's coefficients
    # st.markdown(f"""
    #             Learned coefficients (weights) of the trained model:
    #             <br>
    #             `{ lrmodel_coef_ }`""", unsafe_allow_html=True)

    # model interpretation
    # display the section title
    # st.markdown("#### Model Interpretation")
    # st.markdown("""
    #             <ul>
    #                 <li>The built model's performance is not ideal yet</li>
    #                 <li>Goal: MAE should be below $70,000</li>
    #                 <li>The model features have not yet scaled and the model not tuned</li>
    #             </ul>
    #             """, unsafe_allow_html=True)

    #-------- How Each Feature Affects Price ---------------------

    # display subheader
    # st.subheader("How Each Feature Affects Price")

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
    # st.markdown("##### Feature Coefficients")
    # st.dataframe(coefficients_df)
    
    #-------- Decision Tree Model ---------------------

    # display subheader
    # st.subheader("Decision Tree Model")

    # import decision tree model
    from sklearn.tree import DecisionTreeRegressor

    # Train models
    # lrmodel = func.train_linear_regression(x_train, y_train)
    try:
        lrmodel = func.train_linear_regression(x_train, y_train)
        logging.info("Linear Regression model trained successfully.")
    except Exception as e:
        logging.error(f"Failed to train Linear Regression model: {e}")
        st.error("Error training Linear Regression model.")

    dtmodel = func.train_decision_tree(x_train, y_train)
    try:
        dtmodel = func.train_decision_tree(x_train, y_train)
        logging.info("Decision Tree Regression model trained successfully.")
    except Exception as e:
        logging.error(f"Failed to train Decision Tree Regression model: {e}")
        st.error("Error training Decision Tree Regression model.")

    # Predict and evaluate with lrmodel
    # st.write("Linear Regression Model:")
    # Evaluate on train set
    train_mae = func.evaluate_model(lrmodel, x_train, y_train, dataset_name="Training Set")
    # Evaluate on test set
    test_mae = func.evaluate_model(lrmodel, x_test, y_test, dataset_name="Test Set")

    # Predict and evaluate with dtmodel
    # st.write("Decision Tree Model:")
    # Evaluate on train set
    train_mae_dt = func.evaluate_model(dtmodel, x_train, y_train, dataset_name="Training Set (Decision Tree)")
    # Evaluate on test set
    test_mae_dt = func.evaluate_model(dtmodel, x_test, y_test, dataset_name="Test Set (Decision Tree)")

    #-------- How do I know if my model is Overfitting or Generalized? ---------------------

    # display subheader
    # st.subheader("How do I know if my model is Overfitting or Generalized?")

    # make predictions on train set
    ytrain_pred = dtmodel.predict(x_train)

    # import mean absolute error metric
    from sklearn.metrics import mean_absolute_error

    # evaluate the model
    train_mae = mean_absolute_error(ytrain_pred, y_train)
    # st.write(f"MAE (x_train set): `{train_mae}`")

    #-------- Plot the tree ---------------------

    # display subheader
    # st.subheader("Plot the tree")
    # st.write("Get the features")

    # get the features
    features = dtmodel.feature_names_in_
    # st.write(", ".join(map(str, features)))
    # st.markdown("<br>".join(map(str, features)), unsafe_allow_html=True)

    # display subheader
    # st.subheader("Plot the tree")

    # plot the tree
    import matplotlib.pyplot as plt
    from sklearn import tree

    # Show the plot in Streamlit
    # st.write("Decision Tree Visualization")

    # Plot decision tree
    fig = func.plot_tree_model(dtmodel, dtmodel.feature_names_in_)

    #-------- Random Forest Model ---------------------

    # display subheader
    # st.subheader("Random Forest Model")

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
    # st.write(f"MAE (x_test set): `{test_mae}`")

    #-------- Pickle ---------------------

    # display subheader
    # st.subheader("Pickle:")
    # st.markdown("""
    # - The pickle module implements a powerful algorithm for serializing and de-serializing a Python object structure.
    # - The saving of data is called Serialization, and loading the data is called De-serialization.

    # **Pickle module provides the following functions:**

    # - **pickle.dump** to serialize an object hierarchy, you simply use `dump()`.
    # - **pickle.load** to deserialize a data stream, you call the `load()` function.
    # """)

    # import pickle to save model
    import pickle

    # Save the trained model on the drive
    try:
        pickle.dump(dtmodel, open('RE_Model', 'wb'))
        logging.info("Model saved (pickled) successfully.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        st.error("Error saving the model.")

    # Load the pickled model
    try:
        RE_Model = pickle.load(open('RE_Model', 'rb'))
        logging.info("Model loaded (unpickled) successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        st.error("Error loading the model.")

    # Use the loaded pickled model to make predictions
    RE_Model.predict([[2012, 216, 74, 1, 1, 618, 2000, 600, 1, 0, 0, 6, 0, 1]])

    # st.write("Use the loaded pickled model to make predictions")
    x_test_head_1 = x_test.head(1)
    # st.dataframe(x_test_head_1)
    
    #-------- User Input Form for Predicting House Price ---------------------

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🏠 Predict House Price Based on Your Inputs")

    # Create a form so user has to submit inputs
    with st.form(key="prediction_form"):
        st.markdown("### Enter the property details:")

        # Split the form into two columns
        col1, col2 = st.columns(2)

        with col1:
            year_built = st.number_input("Year Built:", min_value=1800, max_value=2025, value=2015)
            annual_income = st.number_input("Annual Income (in thousands):", min_value=0, max_value=1000, value=100)
            lot_size = st.number_input("Lot Size (sq ft):", min_value=0, max_value=10000, value=600)
            property_type_bunglow = st.selectbox("Is it a Bungalow?", options=[0, 1])
            property_type_condo = st.selectbox("Is it a Condo?", options=[0, 1])
            property_type_duplex = st.selectbox("Is it a Duplex?", options=[0, 1])

        with col2:
            living_area = st.number_input("Living Area (sq ft):", min_value=0, max_value=10000, value=2000)
            basement_area = st.number_input("Basement Area (sq ft):", min_value=0, max_value=5000, value=600)
            has_garage = st.selectbox("Has Garage?", options=[0, 1])
            has_pool = st.selectbox("Has Pool?", options=[0, 1])
            has_fireplace = st.selectbox("Has Fireplace?", options=[0, 1])
            num_bathrooms = st.slider("Number of Bathrooms:", 1, 10, 2)
            has_shed = st.selectbox("Has Shed?", options=[0, 1])
            has_fence = st.selectbox("Has Fence?", options=[0, 1])

        # Submit button
        submit_button = st.form_submit_button(label="Predict House Price")

    # After user clicks submit
    if submit_button:
        try:
            # Make prediction
            user_input = [[
                year_built,
                annual_income,
                lot_size,
                property_type_bunglow,
                property_type_condo,
                property_type_duplex,
                living_area,
                basement_area,
                has_garage,
                has_pool,
                has_fireplace,
                num_bathrooms,
                has_shed,
                has_fence
            ]]

            prediction = RE_Model.predict(user_input)[0]
            
            # Display prediction nicely
            st.success(f"🏠 Predicted House Price: **${prediction:,.2f}**")

        except Exception as e:
            logging.error(f"Prediction failed: {e}", exc_info=True)
            st.error("Prediction failed. Please check your inputs or try again.")

    
except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    st.error("Something went wrong! Please check the logs or try again later.")
