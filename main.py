import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-----------------DATA-PREPARATION-----------------#

data = pd.read_csv('Advertising Budget and Sales.csv')
X = data.iloc[:,-4:-1]
y = data.iloc[:,-1]
#DATA SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 1)
#------------------------LINEAR REGRESSION----------------------------#

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

#------------------------LINEAR REGRESSION----------------------------#

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

#------------------------PREDICTION AND EVALUATION----------------------------#

#RMSA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
y_pred = lr.predict(X_test)
RMSA_VAL = np.sqrt(mean_squared_error(y_test, y_pred))
#R2_SCORE
R2_SCORE = lr.score(X, y)
print("RMSA:" + str(RMSA_VAL) + "\nR2_SCORE:" + str(R2_SCORE))
#Prediction
prediction = lr.predict(pd.DataFrame([[200, 200, 200]], columns=X.columns))
print("Prediction: " + str(round(prediction[0], 2)))


#-------------------------------UI----------------------------------------------#

st.set_page_config(page_title="Advertising Budget and Sales Prediction", page_icon="📊", layout="wide")

# Custom CSS for consistent color scheme and animations
st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css');
    .stApp {
        background-color: #000000;
    }
    .stButton > button {
        background-color: #0080FF;
        color: white !important;
        border: none;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 10px rgba(0,128,255,0.5);
    }
    .stTextInput > div > div > input {
        border-color: #0080FF;
    }
    .stNumberInput > div > div > input {
        border-color: #0080FF;
    }
    h1, h2, h3 {
        background: linear-gradient(45deg, #0080FF, #00FFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    p {
        color: #0080FF;
    }
    .metric-value {
        background: linear-gradient(45deg, #0080FF, #00FFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        font-size: 24px;
        display: inline-block;
    }
    .animate-number {
        display: inline-block;
        animation: fadeInUp 1s ease-out;
    }
    .card {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,128,255,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Advertising Budget and Sales Prediction")
st.subheader("DATA")
st.dataframe(data)
st.subheader("Prepared Data")
st.write("X variable: ")
st.dataframe(X)
st.write("Y variable: ")
st.dataframe(y)
st.subheader("Evaluation ")

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        f"""
        <div class='card'>
            <p>RMSA: <span class='metric-value animate-number'>{RMSA_VAL:.4f}</span></p>
        </div>
        """, 
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"""
        <div class='card'>
            <p>R2 SCORE: <span class='metric-value animate-number'>{R2_SCORE:.4f}</span></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

st.title("Prediction")
st.write("Enter the advertising budget for TV, Radio, and Newspaper:")
tv = st.number_input("TV Budget", min_value=0.0, step=0.1)
radio = st.number_input("Radio Budget", min_value=0.0, step=0.1)
newspaper = st.number_input("Newspaper Budget", min_value=0.0, step=0.1)
predict_button = st.button("Predict", key="predict_button", use_container_width=True)

if predict_button:
    prediction = lr.predict(pd.DataFrame([[tv, radio, newspaper]], columns=X.columns))
    st.markdown(
        f"""
        <div class='card'>
            <p>Predicted Sales: <span class='metric-value animate-number'>${prediction[0]:.2f}</span></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Visualization
st.subheader("Budget Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
budget_data = [tv, radio, newspaper]
labels = ['TV', 'Radio', 'Newspaper']
colors = ['#0080FF', '#00BFFF', '#00FFFF']
ax.pie(budget_data, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax.axis('equal')
st.pyplot(fig)
