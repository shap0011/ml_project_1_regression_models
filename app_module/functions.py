import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error
from sklearn import tree
import streamlit as st

def load_data(filepath):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df

def train_linear_regression(x_train, y_train):
    """Train a Linear Regression model."""
    # create an instance of the LinearRegression model
    model = LinearRegression()
    # fit the model using the training data (input features and target)
    model.fit(x_train, y_train)
    return model

def train_decision_tree(x_train, y_train):
    """Train a Decision Tree Regressor."""
     # create an instance of the DecisionTreeRegressor
    dt = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=567)
    # fit the model using the training data (input features and target)
    dt.fit(x_train, y_train)
    return dt

def evaluate_model(model, x, y, dataset_name="Dataset"):
    """Evaluate a model and return MAE."""
    # make predictions using the train or test set
    y_pred = model.predict(x)
    # evaluate the model and assign to 'mae' variable
    mae = mean_absolute_error(y, y_pred)
    # display result
    st.write(f"MAE ({dataset_name}): `{mae}`")
    return mae

def plot_tree_model(model, feature_names):
    """Plot and display a decision tree."""
    fig, ax = plt.subplots(figsize=(30, 10))
    tree.plot_tree(model, feature_names=feature_names, filled=True, rounded=True, fontsize=10, ax=ax)
    st.pyplot(fig)
    fig.savefig('tree.png', dpi=300)
