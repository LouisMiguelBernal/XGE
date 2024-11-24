import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import base64

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def calculate_metrics(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    return r2, rmse, mae

st.set_page_config(page_title="XGE", layout="wide", page_icon='assets/icon.png')

# Function to convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Convert logo to base64
logo_base64 = get_base64_image('assets/icon.png')


# Display the logo and title using HTML with added margin/padding to move it down
st.markdown(
    f"""
    <div style="display: flex; align-items: center; padding-top: 50px;">
        <img src="data:image/png;base64,{logo_base64}" style="width: 100px; height: auto; margin-right: 10px;">
        <h1 style="margin: 0;">X<span style="color:#0057A0;">GE</span></h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("""
    <style>
    .title {
      font-size: 60px;
      font-family: 'Arial', sans-serif;
      text-align: center;
      margin-bottom: 20px;
    }
    .blue {
        color: #0057A0;  
    }
    .main > div {
        padding-top: 30px;
    }
    .stTabs [role="tablist"] button {
        font-size: 1.2rem;
        padding: 12px 24px;
        margin-right: 10px;
        border-radius: 8px;
        background-color: #0057A0;
        color: white;
    }
    .stTabs [role="tablist"] button:focus, .stTabs [role="tablist"] button[aria-selected="true"] {
        background-color: #0057A0;
        color: white;
    }
    .stTabs [role="tabpanel"] {
        padding-top: 30px;
    }
    .logo-and-name {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .logo-img {
        border-radius: 50%;
        width: 50px;
        height: 50px;
    }                
    </style>
    """, unsafe_allow_html=True)

# Hardcoding the dataset for training the model
@st.cache_data
def load_and_preprocess_data():
    # Load hardcoded car.csv data for training
    df = pd.read_csv("assets/car.csv")  
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(':', '').str.replace('*', '')
    df.rename(columns={'model': 'year', 'make': 'maker', 'model.1': 'model', 'unnamed_9': 'fuel_consumption2',
                       'unnamed_10': 'fuel_consumption3', 'unnamed_11': 'fuel_consumption4'}, inplace=True)

    df.drop(['year', 'fuel_consumption2', 'fuel_consumption3', 'fuel_consumption4'], axis=1, inplace=True)

    categorical_cols = df.select_dtypes('object').columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Split data for prediction
    x = df.drop(['co2_emissions'], axis=1).values
    y = df['co2_emissions'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # RobustScaler
    rc = RobustScaler()
    x_train = rc.fit_transform(x_train)
    x_test = rc.transform(x_test)

    return x_train, x_test, y_train, y_test, rc

# Define functions for training and prediction
def preprocess_and_predict(x_train, x_test, y_train, y_test, rc):
    # Train model (XGBoost in this case)
    xg_params = {
        'objective': ['reg:squarederror'],
        'eta': [0.03, 0.04],  
        'alpha': [10, 12],  
        'lambda': [10, 12],  
        'gamma': [10, 12],  
        'max_depth': [1, 2],  
        'min_child_weight': [2, 4], 
        'subsample': [0.04, 0.5],  
        'colsample_bytree': [0.6, 0.7],  
        'eval_metric': ['rmse']
    }
    
    xg_grid = GridSearchCV(XGBRegressor(), xg_params, cv=5, n_jobs=-1)
    xg_grid.fit(x_train, y_train)
    xgb = xg_grid.best_estimator_

    y_pred = xgb.predict(x_test)
    r2, rmse, mae = calculate_metrics(y_test, y_pred)
    return xgb, r2, rmse, mae

x_train, x_test, y_train, y_test, rc = load_and_preprocess_data()
model, r2, rmse, mae = preprocess_and_predict(x_train, x_test, y_train, y_test, rc)

# Input data for prediction
st.sidebar.header("Input Data For Prediction")
new_file = st.sidebar.file_uploader("", type=["csv"])

predict, model_info = st.tabs(['Prediction','Model Information'])

# Prediction Tab
with predict:
    if new_file is None:
        # Show the title and video when no CSV file is uploaded
        st.markdown("<h1>Drive The <span style='color:#0057A0;'> FUTURE</span></h1>", unsafe_allow_html=True)
        st.video("assets/vehicle.mp4")
    else:
        # Read the original data
        original_df = pd.read_csv(new_file)

        # Make a copy of the original data for preprocessing
        new_df = original_df.copy()

        # Preprocess and scale the new data
        new_df.columns = new_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(':', '').str.replace('*', '')

        # Encode categorical variables
        for col in new_df.select_dtypes('object').columns:
            new_df[col] = LabelEncoder().fit_transform(new_df[col])

        # Ensure the same columns are used as in the training data
        columns_to_use = ['maker', 'model', 'vehicle_class', 'engine_size', 'cylinders', 'transmission', 'fuel', 'fuel_consumption']
        new_df = new_df[columns_to_use]

        # Get only X variables (no 'co2_emissions')
        new_x = new_df.values
        new_x = rc.transform(new_x)  # Apply the same scaling used in training

        # Predict with the trained model
        predictions = model.predict(new_x)

        # Add predictions to the original data
        original_df['Prediction'] = predictions

        # Display the updated dataframe with original values and predictions
        st.title("Prediction Result")
        st.markdown("""<h5>Efficient Predictions at Your Fingertips</h5>""", unsafe_allow_html=True)
        st.dataframe(original_df.head(), width=1800, height=200)

        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)

        with col1:
            # Plot the distribution of the predictions using Plotly
            fig = px.histogram(original_df, x='Prediction', nbins=30, marginal="box", title="Data Prediction Distribution")
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig)

        with col2:
            # Pie chart for 'vehicle_class' using Plotly
            vehicle_class_counts = original_df['vehicle_class'].value_counts()

            # Create pie chart
            fig = go.Figure(data=[go.Pie(labels=vehicle_class_counts.index, values=vehicle_class_counts.values, 
                                        hoverinfo='label+percent', textinfo='percent', hole=0.3)])
            fig.update_layout(title="Vehicle Class Chart")
            st.plotly_chart(fig)

        col1, col2 = st.columns(2)
        with col1:
            top_makers = original_df['maker'].value_counts().nlargest(10)
            fig = px.bar(x=top_makers.index, y=top_makers.values, 
                        title="Manufacturers Distribution", labels={'x': 'Maker', 'y': 'Count'})
            st.plotly_chart(fig)

        with col2:
            fig = px.scatter(original_df, x='engine_size', y='fuel_consumption', color='vehicle_class',
                    title="Engine Size vs Fuel Consumption",
                    labels={'engine_size': 'Engine Size (L)', 'fuel_consumption': 'Fuel Consumption (L/100km)'})
            st.plotly_chart(fig)


# Model Information Tab
with model_info:
    st.title('XGBoost')
    st.markdown("""<h5>The boosting algorithm also known as Extreme Gradient Boosting (XGBoost) is a powerful machine learning algorithm. 
                The XGE is developed and tuned by key features of maker, model, vehicle class, engine size, cylinders, transmission, fuel, and fuel consumption. 
                It builds upon the principles of gradient boosting, combining multiple weak learners—typically decision trees—into a strong predictive model.
               </h5>""", unsafe_allow_html=True)

    model_df = pd.read_csv("assets/car.csv")  
    model_df.columns = model_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(':', '').str.replace('*', '')
    model_df.rename(columns={'model': 'year', 'make': 'maker', 'model.1': 'model', 'unnamed_9': 'fuel_consumption2',
                       'unnamed_10': 'fuel_consumption3', 'unnamed_11': 'fuel_consumption4'}, inplace=True)
    model_df.drop(['year', 'fuel_consumption2', 'fuel_consumption3', 'fuel_consumption4'], axis=1, inplace=True)

    # Create the R² Donut Chart
    fig_r2 = go.Figure(go.Pie(
        values=[r2, 1 - r2],
        labels=["R² Score", ""],
        hole=0.7,
        marker=dict(colors=["#00704A", "#e6e6e6"]),
        textinfo='none'
    ))
    fig_r2.add_annotation(
        text=f"{r2 * 100:.0f}%",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=33, color="#00704A", weight = 'bold')
    )
    fig_r2.update_layout(
        title="Excellent Model Fit: High R² Score",
        margin=dict(t=40, b=10, l=10, r=10),  
        width=250,
        height=250
    )

    # Create the RMSE and MAE Bar Charts
    fig_bars = go.Figure()
    fig_bars.add_trace(go.Bar(
        y=["RMSE"],
        x=[rmse],
        orientation='h',
        marker=dict(color='#00704A'),
        name="RMSE"
    ))
    fig_bars.add_trace(go.Bar(
        y=["MAE"],
        x=[mae],
        orientation='h',
        marker=dict(color='#00704A'),
        name="MAE"
    ))

    fig_bars.update_xaxes(range=[0, 100], showgrid=True)
    fig_bars.update_yaxes(showgrid=False)
    fig_bars.update_layout(
        title="Regression Model Accuracy: Low RMSE & MAE",
        barmode='stack',
        margin=dict(t=40, b=10, l=10, r=10),  # Increased top margin
        height=250,
        width=250,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)'
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_r2, use_container_width=True)

    with col2:
        st.plotly_chart(fig_bars, use_container_width=True)
    with col1:
        # Feature Importance bar chart from XGBoost model
        importances = model.feature_importances_  #
        feature_names = model_df.columns[:-1]  

        fig = px.bar(x=feature_names, y=importances, 
                    labels={'x': 'Features', 'y': 'Importance'},
                    title="XGBoost Feature Importance")
        st.plotly_chart(fig)
    with col2:
        # Pie chart for 'vehicle_class' using Plotly
        vehicle_class_counts = model_df['vehicle_class'].value_counts().nlargest(5)

        # Create pie chart
        fig = go.Figure(data=[go.Pie(labels=vehicle_class_counts.index, values=vehicle_class_counts.values, 
                                    hoverinfo='label+percent', textinfo='percent', hole=0.3)])
        fig.update_layout(title="Vehicle Class Chart")
        st.plotly_chart(fig)

    # top 10 manufacturers chart
    top_10_makers = model_df['maker'].value_counts().nlargest(10)

    fig = px.bar(top_10_makers, y=top_10_makers.index, x=top_10_makers.values,
                labels={'x': 'Count'},
                title='Top 10 Car Manufacturers',
                color=top_10_makers.index) # You can color by manufacturer
    st.plotly_chart(fig)



# Footer content with logo as base64
footer = f"""
<hr>
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; padding: 10px 0;">
  <div style="flex-grow: 1; text-align: left;">
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" style="width: 100px; margin-right: 10px;">
        <h1 style="margin: 0;">X<span style="color:#0057A0;">GE</span></h1>
    </div>
  </div>
  <!-- Copyright -->
  <div style="flex-grow: 1; text-align: center;">
    <span>Copyright 2024 | All Rights Reserved</span>
  </div>
  <!-- Social media icons -->
  <div style="flex-grow: 1; text-align: right;">
    <a href="https://www.linkedin.com" class="fa fa-linkedin" style="padding: 10px; font-size: 24px; background: #0077B5; color: white; text-decoration: none; margin: 5px;"></a>
    <a href="https://www.instagram.com" class="fa fa-instagram" style="padding: 10px; font-size: 24px; background: #E1306C; color: white; text-decoration: none; margin: 5px;"></a>
    <a href="https://www.youtube.com" class="fa fa-youtube" style="padding: 10px; font-size: 24px; background: #FF0000; color: white; text-decoration: none; margin: 5px;"></a>
    <a href="https://www.facebook.com" class="fa fa-facebook" style="padding: 10px; font-size: 24px; background: #3b5998; color: white; text-decoration: none; margin: 5px;"></a>
    <a href="https://twitter.com" class="fa fa-twitter" style="padding: 10px; font-size: 24px; background: #1DA1F2; color: white; text-decoration: none; margin: 5px;"></a>
  </div>
</div>
"""

# Display footer
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">', unsafe_allow_html=True)
st.markdown(footer, unsafe_allow_html=True)
