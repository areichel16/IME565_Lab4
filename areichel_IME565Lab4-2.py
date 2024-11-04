# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Mobile Classifier: A Machine Learning App') 
st.write("This app uses 20 inputs to predict the cost range of your mobile device "
         "built on the Mobile dataset using either Decision Tree or Regression Forest. Use the inputs in the sidebar to "
         "make your prediction!")

# Display an image of penguins
st.image('mobile_image.jpg', width = 400)

with open('decision_tree_mobile.pickle', 'rb') as dt_pickle:
    dt_model = pickle.load(dt_pickle)
with open('random_forest_mobile.pickle', 'rb') as rf_pickle:
    rf_model = pickle.load(rf_pickle)

# Create a sidebar for input collection
st.sidebar.header('Mobile Features Input')
battery_power = st.sidebar.number_input("Enter your mobile's battery power", step=1)
blue = st.sidebar.selectbox('Bluetooth', options=['Yes', 'No'])
clock_speed = st.sidebar.number_input("Enter your mobile's clock speed (GHz)", step=0.1)
dual_sim = st.sidebar.selectbox('Dual SIM', options=['Yes', 'No'])
fc = st.sidebar.number_input("Enter your mobile's front camera (MP)", step=1)
four_g = st.sidebar.selectbox('4G Support', options=['Yes', 'No'])
int_memory = st.sidebar.number_input("Enter your mobile's internal memory (GB)", step=1)
m_dep = st.sidebar.number_input("Enter your mobile's depth (cm)", step=0.1)
mobile_wt = st.sidebar.number_input("Enter your mobile's weight (g)", step=1)
n_cores = st.sidebar.number_input("Enter your mobile's number of cores", step=1)
pc = st.sidebar.number_input("Enter your mobile's primary camera (MP)", step=1)
px_height = st.sidebar.number_input("Enter your mobile's pixel height", step=1)
px_width = st.sidebar.number_input("Enter your mobile's pixel width", step=1)
ram = st.sidebar.number_input("Enter your mobile's RAM (MB)", step=1)
sx_h = st.sidebar.number_input("Enter your mobile's screen height (cm)", step=1)
sx_w = st.sidebar.number_input("Enter your mobile's screen width (cm)", step=1)
talk_time = st.sidebar.number_input("Enter your mobile's talk time (min)", step=1)
three_g = st.sidebar.selectbox('3G Support', options=['Yes', 'No'])
touch_screen = st.sidebar.selectbox('Touch Screen', options=['Yes', 'No'])
wifi = st.sidebar.selectbox('WiFi Support', options=['Yes', 'No'])

# User selects model 
model_choice = st.sidebar.selectbox('Select Model for Prediction', options=['Decision Tree', 'Random Forest'])

if st.sidebar.button("Predict Price Range"):
    # Prepare input data
    input_data = np.array([[
        battery_power,
        1 if blue == 'Yes' else 0,
        clock_speed,
        1 if dual_sim == 'Yes' else 0,
        fc,
        1 if four_g == 'Yes' else 0,
        int_memory,
        m_dep,
        mobile_wt,
        n_cores,
        pc,
        px_height,
        px_width,
        ram,
        sx_h,
        sx_w,
        talk_time,
        1 if three_g == 'Yes' else 0,
        1 if touch_screen == 'Yes' else 0,
        1 if wifi == 'Yes' else 0
    ]])

    # Make prediction based on the selected model
    if model_choice == 'Decision Tree':
        predicted_class = dt_model.predict(input_data)
        predicted_prob = dt_model.predict_proba(input_data)    
        class_labels = dt_model.classes_  # Get class labels
    else:  # Random Forest
        predicted_class = rf_model.predict(input_data)
        predicted_prob = rf_model.predict_proba(input_data)
        class_labels = rf_model.classes_  # Get class labels

    # Get the predicted class index
    predicted_class_index = np.where(class_labels == predicted_class[0])[0][0]

    # Display the prediction results
    st.write("### Prediction Results")
    st.write(f"**Predicted Price Range Class:** {predicted_class[0]}")
    st.write(f"**Probability of the Predicted Class:** {predicted_prob[0][predicted_class_index] * 100:.2f}%")
    
    # Showing additional items in tabs
    st.subheader("Prediction Performance")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])
    
    if model_choice == 'Decision Tree': 
        # Tab 1: Feature Importance Visualization
        with tab1:
            st.write("### Feature Importance")
            st.image('feature_imp2.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 2: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('confusion_mat2.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 3: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('class_report2.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")
    
    else: # Random Forest
        # Tab 1: Feature Importance Visualization
        with tab1:
            st.write("### Feature Importance")
            st.image('feature_imp3.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 2: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('confusion_mat3.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 3: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('class_report3.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")