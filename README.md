# House Price Prediction App

This project is a **Streamlit web application** that builds and evaluates several **machine learning regression models** to predict **house prices**.

## Features

- Load and preview the dataset
- Train a **Linear Regression** model
- Train a **Decision Tree Regressor**
- Train a **Random Forest Regressor**
- Evaluate models using **Mean Absolute Error (MAE)**
- Save and load a trained model using **Pickle**
- Display logs and error messages clearly
- Modularized code with reusable functions

## Technologies Used

- [Streamlit](https://streamlit.io/) - For building the interactive web app
- [Scikit-learn](https://scikit-learn.org/) - For machine learning models
- [Pandas](https://pandas.pydata.org/) - For data manipulation
- [Pickle](https://docs.python.org/3/library/pickle.html) - For saving and loading models
- [Logging](https://docs.python.org/3/library/logging.html) - For backend log management

## Project Structure

- **.streamlit/**
  - `config.toml` — Theme setting
- `regression_app.py` — Main Streamlit app
- **app_module/**
  - `__init__.py`
  - `functions.py` — All helper functions
- **data/**
  - `final.csv` — Raw dataset
- `RE_Model` — Pickled trained model (saved Decision Tree)
- `requirements.txt` — List of Python dependencies
- `README.md` — Project documentation

## How to Run the App Locally

1. **Clone the repository**

```bash```
git clone https://github.com/shap0011/ml_project_1_regression_models.git
cd ml_project_1_regression_models

2. **Install the required packages**

```bash```
    pip install -r requirements.txt

3. **Run the App**

```bash```
streamlit run regression_app.py

4. Open the URL shown (usually http://localhost:8501) to view the app in your browser!

## Deployment
The app is also deployed on Streamlit Cloud.
Click [![Here](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://home-price-prediction-app-shap0011.streamlit.app/) to view the live app.

## Model Details

| Model                    | Description                        |
|--------------------------|------------------------------------|
| Linear Regression        | Baseline regression model          |
| Decision Tree Regressor  | Tree-based regression model        |
| Random Forest Regressor  | Ensemble of Decision Trees         |

Evaluation metric used: **Mean Absolute Error (MAE)**

## Logging and Error Handling

Backend logging is enabled using Python’s logging module.

Important messages are displayed inside the app using Streamlit's st.success(), st.warning(), st.error().

The app is robust against missing files or bad data.

## Author
Name: Olga Durham

LinkedIn: [\[Olga Durham LinkedIn Link\]](https://www.linkedin.com/in/olga-durham/)

GitHub: [\[Olga Durham GitHub Link\]](https://github.com/shap0011)

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://refactored-space-tribble-5rv6wwwgx74cv75p.github.dev/)

## License

This project is licensed under the MIT License.  
Feel free to use, modify, and share it.  
See the [LICENSE](./LICENSE) file for details.