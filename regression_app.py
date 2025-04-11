import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'info'

import logging

# Configure logging manually again
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

import streamlit as st
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from app_module import functions as func

# Load CSV only once
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# Load pre-trained model only once
@st.cache_resource
def load_trained_model(path):
    return pickle.load(open(path, 'rb'))

try:  
    
    #-------- page setting, header, intro ---------------------
      
    # Set page configuration
    st.set_page_config(page_title="🏡 House Price Prediction App", layout="centered")
    # set the title of the Streamlit app    
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
    
    #-------- the linear regression model ---------------------
    
    # separate the input features (all columns except 'price')
    x = df.drop('price', axis=1)

    # store the target variable (price) in y
    y = df['price']
    
    #-------- train-test split ---------------------

    # first attempt: split the data randomly (20% test, 80% train)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

    # count and display how many 'Bungalow' type properties are in the training set
    # this helps check if the class distribution is balanced
    x_train_bungalow_random = x_train.property_type_Bunglow.value_counts()
 
    # second attempt: split the data while preserving the distribution of 'Bungalow' property type
    # stratify ensures proportional representation in both training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=x.property_type_Bunglow)

    # count and display 'Bungalow' values again after stratified split to compare distribution
    x_train_bungalow_stratified = x_train.property_type_Bunglow.value_counts()

    # Train model
    try:
        lrmodel = func.train_linear_regression(x_train, y_train)
        logging.info("Linear Regression model trained successfully.")
    except Exception as e:
        logging.error(f"Failed to train Linear Regression model: {e}")
        st.error("Error training Linear Regression model.")
    
    #-------- Make Predictions and Evaluate Model Performance ---------------------

    # making predictions and evaluating model performance

    # predict house prices on the training set using the trained model
    train_pred = lrmodel.predict(x_train)

    # calculate MAE between predicted and actual house prices
    train_mae = mean_absolute_error(train_pred, y_train)
  
    #-------- Decision Tree Model ---------------------

    # Train models
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

    # Evaluate on train set
    train_mae = func.evaluate_model(lrmodel, x_train, y_train, dataset_name="Training Set")
    # Evaluate on test set
    test_mae = func.evaluate_model(lrmodel, x_test, y_test, dataset_name="Test Set")

    # Predict and evaluate with dtmodel
    # Evaluate on train set
    train_mae_dt = func.evaluate_model(dtmodel, x_train, y_train, dataset_name="Training Set (Decision Tree)")
    # Evaluate on test set
    test_mae_dt = func.evaluate_model(dtmodel, x_test, y_test, dataset_name="Test Set (Decision Tree)")

    #-------- How do I know if my model is Overfitting or Generalized? ---------------------

    # make predictions on train set
    ytrain_pred = dtmodel.predict(x_train)

    # evaluate the model
    train_mae = mean_absolute_error(ytrain_pred, y_train)
    # st.write(f"MAE (x_train set): `{train_mae}`")

    #-------- Random Forest Model ---------------------

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

    #-------- Pickle ---------------------

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
    # RE_Model.predict([[2012, 216, 74, 1, 1, 618, 2000, 600, 1, 0, 0, 6, 0, 1]])
    
    #-------- User Input Form for Predicting House Price ---------------------

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🏠 Predict House Price Based on Your Inputs")

    # Feature names expected by the model
    feature_names = [
        'year_sold', 'property_tax', 'insurance', 'beds', 'baths',
        'sqft', 'year_built', 'lot_size', 'basement', 
        'popular', 'recession', 'property_age', 
        'property_type_Bunglow', 'property_type_Condo'
    ]

    # Create a form so user has to submit inputs
    with st.form(key="prediction_form"):
        st.markdown("### Enter the property details:")

        # Form input
        col1, col2 = st.columns(2)

        with col1:
            year_sold = st.number_input("Year Sold:", min_value=1990, max_value=2025, value=2022)
            property_tax = st.number_input("Property Tax ($):", min_value=0, value=2000)
            insurance = st.number_input("Insurance ($):", min_value=0, value=1500)
            beds = st.number_input("Number of Bedrooms:", min_value=0, value=3)
            baths = st.number_input("Number of Bathrooms:", min_value=0, value=2)
            sqft = st.number_input("Living Area (sq ft):", min_value=0, value=2000)

        with col2:
            year_built = st.number_input("Year Built:", min_value=1800, max_value=2025, value=2005)
            lot_size = st.number_input("Lot Size (sq ft):", min_value=0, value=5000)
            basement = st.number_input("Basement Area (sq ft):", min_value=0, value=500)
            popular = st.selectbox("Is it in a Popular Area?", options=[0, 1])
            recession = st.selectbox("Was it sold during a Recession?", options=[0, 1])
            property_type_Bunglow = st.selectbox("Is it a Bungalow?", options=[0, 1])
            property_type_Condo = st.selectbox("Is it a Condo?", options=[0, 1])

        submit_button = st.form_submit_button(label="Predict House Price")

    # 🔥 Move this inside submit_button
    if submit_button:
        try:
            # Calculate property age dynamically after user input
            property_age = year_sold - year_built

            # Display the calculated property age
            st.markdown(f"🧮 Property Age (calculated): **{property_age}** years")

            # Create the DataFrame for prediction
            user_input = pd.DataFrame([[
                year_sold, property_tax, insurance, beds, baths, sqft,
                year_built, lot_size, basement, popular, recession, property_age,
                property_type_Bunglow, property_type_Condo
            ]], columns=feature_names)

            prediction = RE_Model.predict(user_input)[0]
            st.success(f"🏠 Predicted House Price: **${prediction:,.2f}**")

        except Exception as e:
            logging.error(f"Prediction failed: {e}", exc_info=True)
            st.error("Prediction failed. Please check your inputs or try again.")
    
except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    st.error("Something went wrong! Please check the logs or try again later.")
