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

# Function to calculate metrics
def calculate_metrics(model, x_test, y_test):
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    return r2, rmse, mae

# Load and preprocess data function
def load_and_preprocess_data(apply_iqr=False):
    df = pd.read_csv("assets/car.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(':', '').str.replace('*', '')
    df.rename(columns={
        'model': 'year', 'make': 'maker', 'model.1': 'model', 
        'unnamed_9': 'fuel_consumption2', 'unnamed_10': 'fuel_consumption3', 'unnamed_11': 'fuel_consumption4'
    }, inplace=True)
    df.drop(['year', 'fuel_consumption2', 'fuel_consumption3', 'fuel_consumption4'], axis=1, inplace=True)

    # Encode categorical variables
    categorical_cols = df.select_dtypes('object').columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Apply IQR filtering if requested
    if apply_iqr:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # Split data
    x = df.drop(['co2_emissions'], axis=1)
    y = df['co2_emissions']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, scaler

# Train and evaluate the model function
def train_and_evaluate(x_train, x_test, y_train, y_test):
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
    grid = GridSearchCV(XGBRegressor(), xg_params, cv=5, n_jobs=-1)
    grid.fit(x_train, y_train)
    return grid.best_estimator_

# Load and preprocess data
x_train_no_iqr, x_test_no_iqr, y_train_no_iqr, y_test_no_iqr, scaler = load_and_preprocess_data(apply_iqr=False)
model_no_iqr = train_and_evaluate(x_train_no_iqr, x_test_no_iqr, y_train_no_iqr, y_test_no_iqr)

x_train_iqr, x_test_iqr, y_train_iqr, y_test_iqr, _ = load_and_preprocess_data(apply_iqr=True)
model_iqr = train_and_evaluate(x_train_iqr, x_test_iqr, y_train_iqr, y_test_iqr)

# Calculate metrics for models
r2_no_iqr, rmse_no_iqr, mae_no_iqr = calculate_metrics(model_no_iqr, x_test_no_iqr, y_test_no_iqr)
r2_iqr, rmse_iqr, mae_iqr = calculate_metrics(model_iqr, x_test_iqr, y_test_iqr)

# Input data for prediction
st.sidebar.header("Input Data For Prediction")
new_file = st.sidebar.file_uploader("", type=["csv"])

predict, model_info = st.tabs(['Prediction','Model Information'])

# Prediction Tab
with predict:
    if new_file is None:
        st.markdown("<h1>Drive The <span style='color:#0057A0;'> FUTURE</span></h1>", unsafe_allow_html=True)
        st.video("assets/vehicle.mp4")
    else:
        # Read the uploaded file
        original_df = pd.read_csv(new_file)

        # --- Clean Column Names ---
        cleaned_df = original_df.copy()
        cleaned_df.columns = (
            cleaned_df.columns.str.strip()
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(':', '')
            .str.replace('*', '')
        )

        # --- Encode Categorical Variables ---
        for col in cleaned_df.select_dtypes('object').columns:
            cleaned_df[col] = LabelEncoder().fit_transform(cleaned_df[col])

        # --- Specify Required Columns ---
        columns_to_use = [
            'maker', 'model', 'vehicle_class', 'engine_size',
            'cylinders', 'transmission', 'fuel', 'fuel_consumption'
        ]

        # --- Without IQR ---
        # Create a copy for the 'Without IQR' table
        new_df_no_iqr = cleaned_df[columns_to_use].copy()

        # Scale and Predict
        new_x_no_iqr = scaler.transform(new_df_no_iqr)
        predictions_no_iqr = model_no_iqr.predict(new_x_no_iqr)

        # Add Predictions to Original DataFrame (Without IQR)
        original_df_no_iqr = original_df.copy()
        original_df_no_iqr['Prediction (No IQR)'] = predictions_no_iqr

        # Display the First Table
        st.title("Prediction Result (Without IQR)")
        st.dataframe(original_df_no_iqr.head(), width=1800, height=200)

        # --- With IQR ---
        # Start Fresh for IQR Filtering
        new_df_iqr = cleaned_df[columns_to_use].copy()

        # Apply IQR Filtering
        numeric_cols = new_df_iqr.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            Q1 = new_df_iqr[col].quantile(0.25)
            Q3 = new_df_iqr[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            new_df_iqr = new_df_iqr[(new_df_iqr[col] >= lower_bound) & (new_df_iqr[col] <= upper_bound)]

        # Scale and Predict with IQR
        new_x_iqr = scaler.transform(new_df_iqr)
        predictions_iqr = model_iqr.predict(new_x_iqr)

        # Map Back to Original DataFrame (IQR Filtered Rows Only)
        original_df_iqr_filtered = original_df.loc[new_df_iqr.index].copy()
        original_df_iqr_filtered['Prediction (With IQR)'] = predictions_iqr

        # Display the Second Table
        st.title("Prediction Result (With IQR)")
        st.dataframe(original_df_iqr_filtered.head(), width=1800, height=200)

        st.markdown("""<h2 style="margin: 0;">Comparative<span style="color:#0057A0;"> Analysis</span></h2>""",unsafe_allow_html=True)
        st.markdown("""<h5>The comparison between the predictions without IQR filtering and those with IQR filtering shows that the model without IQR filtering produces results very close to the actual values. 
                    While the IQR-filtered predictions exhibit minimal deviations, these differences are marginal, indicating that the original model's performance is robust. 
                    Despite IQR filtering being useful for handling outliers, its impact on the prediction accuracy here is subtle, suggesting that the data may already be well-balanced or that the model can handle minor variations effectively.
                    </h5>""", unsafe_allow_html=True)

# Model Information Tab
with model_info:
    st.title('XGBoost')
    st.markdown("""<h5>The boosting algorithm also known as Extreme Gradient Boosting (XGBoost) is a powerful machine learning algorithm. 
                The XGE is developed and tuned by key features of maker, model, vehicle class, engine size, cylinders, transmission, fuel, and fuel consumption. 
                It builds upon the principles of gradient boosting, combining multiple weak learners—typically decision trees—into a strong predictive model.
               </h5>""", unsafe_allow_html=True)

        # Create R² Donut Chart
    def create_r2_donut_chart(r2, title, is_with_iqr=False):
        # Set color to red if it's the 'With IQR' chart
        color = "#FF0000" if is_with_iqr else "#0057A0"
        
        fig_r2 = go.Figure(go.Pie(
            values=[r2, 1 - r2],
            labels=["R² Score", ""],
            hole=0.7,
            marker=dict(colors=[color, "#e6e6e6"]),  # Apply red or green color
            textinfo='none'
        ))
        fig_r2.add_annotation(
            text=f"{r2 * 100:.0f}%",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=33, color=color, weight='bold')  # Color the annotation text accordingly
        )
        fig_r2.update_layout(
            title=title,
            margin=dict(t=40, b=10, l=10, r=10),
            width=600,  # Set width to allow charts to expand fully
            height=300
        )
        return fig_r2

    # Create RMSE and MAE Bar Chart
    def create_error_bar_chart(rmse, mae, title, is_with_iqr=False):
        # Set color to red if it's the 'With IQR' chart
        color = "#FF0000" if is_with_iqr else "#0057A0"
        
        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            y=["RMSE"],
            x=[rmse],
            orientation='h',
            marker=dict(color=color),  # Apply red or green color
            name="RMSE"
        ))
        fig_bars.add_trace(go.Bar(
            y=["MAE"],
            x=[mae],
            orientation='h',
            marker=dict(color=color),  # Apply red or green color
            name="MAE"
        ))
        fig_bars.update_xaxes(range=[0, max(rmse, mae) * 1.2], showgrid=True)
        fig_bars.update_yaxes(showgrid=False)
        fig_bars.update_layout(
            title=title,
            barmode='stack',
            margin=dict(t=40, b=10, l=10, r=10),
            height=300,
            width=600,  # Set width to allow charts to expand fully
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig_bars

    # Display Visualizations for Model Performance
    st.header("Model Performance Visualization")

    # Without IQR (Keep Default Green)
    r2_no_iqr, rmse_no_iqr, mae_no_iqr = calculate_metrics(model_no_iqr, x_test_no_iqr, y_test_no_iqr)
    fig_r2_no_iqr = create_r2_donut_chart(r2_no_iqr, "R² Score (Without IQR)", is_with_iqr=False)
    fig_bars_no_iqr = create_error_bar_chart(rmse_no_iqr, mae_no_iqr, "Errors (Without IQR)", is_with_iqr=False)

    # With IQR (Red Color)
    r2_iqr, rmse_iqr, mae_iqr = calculate_metrics(model_iqr, x_test_iqr, y_test_iqr)
    fig_r2_iqr = create_r2_donut_chart(r2_iqr, "R² Score (With IQR)", is_with_iqr=True)
    fig_bars_iqr = create_error_bar_chart(rmse_iqr, mae_iqr, "Errors (With IQR)", is_with_iqr=True)

    # Create a wider layout with columns for "Without IQR" and "With IQR" (full horizontal expansion)
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_r2_no_iqr, use_container_width=True)
        st.plotly_chart(fig_bars_no_iqr, use_container_width=True)

    with col2:
        st.plotly_chart(fig_r2_iqr, use_container_width=True)
        st.plotly_chart(fig_bars_iqr, use_container_width=True)
    
    st.markdown("""<h2 style="margin: 0;">Metrics<span style="color:#0057A0;"> Comparison</span></h2>""",unsafe_allow_html=True)
    st.markdown("""<h5>The XGBoost model's performance before and after applying IQR filtering shows a slight decrease in overall accuracy. 
                Without IQR filtering, the model achieved an R² of 0.9467, an RMSE of 13.489, and an MAE of 9.5. 
                After applying IQR filtering, the R² slightly decreased to 0.9463, while the RMSE increased to 14.03 and the MAE to 10.23. 
                This indicates that while IQR filtering removed potential outliers, it also may have reduced the model's predictive precision, suggesting that the filtered data might have lost some valuable information for accurate prediction. 
                Balancing outlier management with model performance will be crucial for optimizing future iterations.</h5>""", unsafe_allow_html=True)
    
    # For data analysis of charts used to train model
    model_df = pd.read_csv("assets/car.csv")  
    model_df.columns = model_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(':', '').str.replace('*', '')
    model_df.rename(columns={'model': 'year', 'make': 'maker', 'model.1': 'model', 'unnamed_9': 'fuel_consumption2',
                       'unnamed_10': 'fuel_consumption3', 'unnamed_11': 'fuel_consumption4'}, inplace=True)
    model_df.drop(['year', 'fuel_consumption2', 'fuel_consumption3', 'fuel_consumption4'], axis=1, inplace=True)


    top_10_makers = model_df['maker'].value_counts().nlargest(10)
    fig_top_makers = px.bar(
        top_10_makers,
        x=top_10_makers.values,
        y=top_10_makers.index,
        orientation='h',
        title='Top 10 Car Manufacturers',
        labels={'x': 'Number of Cars', 'y': 'Manufacturer'},
        color=top_10_makers.index
    )
    fig_top_makers.update_layout(xaxis_title='Count', yaxis_title='Manufacturer')
    st.plotly_chart(fig_top_makers)

    # Vehicle Class Pie Chart
    vehicle_class_counts = model_df['vehicle_class'].value_counts().nlargest(5)
    fig_vehicle_class = go.Figure(data=[go.Pie(
        labels=vehicle_class_counts.index,
        values=vehicle_class_counts.values,
        hoverinfo='label+percent',
        textinfo='label+percent',  # Display both label and percentage
        marker=dict(colors=px.colors.qualitative.Plotly)  # Distinct colors for each segment
    )])
    fig_vehicle_class.update_layout(title="Top 5 Vehicle Classes")
    st.plotly_chart(fig_vehicle_class)


    

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
