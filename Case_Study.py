import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# File paths (update the paths to your local files)
email_file = "C:\\Users\\ranja\\OneDrive\\Desktop\\Ranjith\\Internship\\Quantacus\\email_table.csv"
opened_file = "C:\\Users\\ranja\\OneDrive\\Desktop\\Ranjith\\Internship\\Quantacus\\email_opened_table.csv"
clicked_file = "C:\\Users\\ranja\\OneDrive\\Desktop\\Ranjith\\Internship\\Quantacus\\link_clicked_table.csv"

# Function to merge data
def preprocess_data():
    email_df = pd.read_csv(email_file)
    opened_df = pd.read_csv(opened_file)
    clicked_df = pd.read_csv(clicked_file)

    opened_df['opened'] = 1
    clicked_df['clicked'] = 1

    email_df = pd.merge(email_df, opened_df[['email_id', 'opened']], on='email_id', how='left')
    email_df = pd.merge(email_df, clicked_df[['email_id', 'clicked']], on='email_id', how='left')

    email_df['opened'] = email_df['opened'].fillna(0).astype(int)
    email_df['clicked'] = email_df['clicked'].fillna(0).astype(int)

    return email_df

# Function to train model
def train_model(email_df):
    cat_cols = ['email_text', 'email_version', 'weekday', 'user_country']
    encoder = LabelEncoder()
    for col in cat_cols:
        email_df[col] = encoder.fit_transform(email_df[col])

    X = email_df.drop(columns=['email_id', 'opened', 'clicked'])
    y = email_df['clicked']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "click_prediction_model.pkl")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc, y_test, y_pred

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'page' not in st.session_state:
    st.session_state.page = "Login"

# Sidebar navigation
if not st.session_state.logged_in:
    page = st.session_state.page
else:
    page = st.sidebar.radio("Choose a page", [
        "Home",
        "Model Evaluation",
        "CTR Analysis",
        "📊 Campaign Insights & Predictions"
    ])

# Login Page
if page == "Login":
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.session_state.logged_in = True
            st.session_state.page = "Home"
            st.experimental_rerun()
        else:
            st.session_state.page = "Sign Up"
            st.experimental_rerun()

# Sign Up Page
elif page == "Sign Up":
    st.title("Sign Up Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if password == confirm_password:
            st.session_state.logged_in = True
            st.session_state.page = "Home"
            st.experimental_rerun()
        else:
            st.error("Passwords do not match")

# Home Page
elif page == "Home":
    st.title("📩 Email Campaign Click-Through Prediction App")
    st.markdown("""
    ## About the App
    This app predicts whether a user will click a link in an email, helping marketers optimize email campaigns by targeting the most engaging users. 
    By using machine learning, we can predict the likelihood that a user will click on a link within an email based on different features.

    ## Features
    - **Click Prediction**: The app predicts whether an email recipient will click on the email link.
    - **Campaign Optimization**: Marketers can improve their CTR (Click-Through Rate) by targeting the right users based on predicted probabilities.
    - **Insights**: The app provides insights into the performance of campaigns based on user behavior, such as which days of the week see higher engagement and which versions of emails perform better.

    ## Key Features of the App:
    - **Data Preprocessing**: The app merges email data with information about whether the email was opened or the link was clicked.
    - **Model Training**: The app uses a **Random Forest Classifier** to predict the likelihood of a user clicking the link in the email.
    - **CTR Analysis**: Analyze the **Click-Through Rate (CTR)** and compare the baseline CTR with the optimized CTR using machine learning-based predictions.
    - **Campaign Insights**: Get insights into factors like email text, user location, and time of the week that influence click rates.

    ## How it works:
    1. The app uses historical email campaign data, which includes whether an email was opened and if a link was clicked.
    2. A **Random Forest model** is trained on these features and used to predict whether future recipients will click on the link in an email.
    3. You can analyze the **Click-Through Rate (CTR)** before and after applying the model's optimization strategy.

    """)
    email_df = preprocess_data()
    st.markdown(f"✅ **Opened Email:** {email_df['opened'].mean() * 100:.2f}%")
    st.markdown(f"✅ **Clicked Link:** {email_df['clicked'].mean() * 100:.2f}%")

# Model Evaluation Page
elif page == "Model Evaluation":
    st.title("🔍 Model Evaluation")
    email_df = preprocess_data()
    model, acc, y_test, y_pred = train_model(email_df)

    st.write(f"**Accuracy:** {acc:.2f}")
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.text("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

# CTR Analysis Page
elif page == "CTR Analysis":
    st.title("📈 Click-Through Rate (CTR) Analysis")
    email_df = preprocess_data()
    model, _, _, _ = train_model(email_df)

    predicted_probs = model.predict_proba(email_df.drop(columns=['email_id', 'opened', 'clicked']))[:, 1]
    email_df['predicted_click_prob'] = predicted_probs

    top_30_percent = email_df.sort_values(by='predicted_click_prob', ascending=False).head(int(len(email_df) * 0.3))

    improved_ctr = top_30_percent['clicked'].mean() * 100
    baseline_ctr = email_df['clicked'].mean() * 100

    st.write(f"🚀 **Baseline CTR:** {baseline_ctr:.2f}%")
    st.write(f"🎯 **Optimized CTR (Top 30% strategy):** {improved_ctr:.2f}%")
    st.write(f"📊 **CTR Improvement:** {improved_ctr - baseline_ctr:.2f}%")

    def plot_bar(data, x, y, title):
        plt.figure(figsize=(8, 4))
        sns.barplot(data=data, x=x, y=y, estimator=np.mean)
        plt.title(title)
        st.pyplot(plt)

    plot_bar(email_df, 'weekday', 'clicked', 'Click Rate by Weekday')
    plot_bar(email_df, 'email_version', 'clicked', 'Click Rate by Email Version')
    plot_bar(email_df, 'user_country', 'clicked', 'Click Rate by Country')

# Campaign Insights & Predictions Page
elif page == "📊 Campaign Insights & Predictions":
    st.title("📊 Campaign Insights & Predictions")

    email_df = preprocess_data()
    model, acc, _, _ = train_model(email_df)
    predicted_probs = model.predict_proba(email_df.drop(columns=['email_id', 'opened', 'clicked']))[:, 1]
    email_df['predicted_click_prob'] = predicted_probs

    opened_pct = email_df['opened'].mean() * 100
    clicked_pct = email_df['clicked'].mean() * 100

    st.subheader("📌 What percentage of users opened the email and what percentage clicked on the link?")
    st.write(f"- **Opened Emails:** {opened_pct:.2f}%")
    st.write(f"- **Clicked Links:** {clicked_pct:.2f}%")

    st.subheader("📌 Can you build a model to optimize email sending?")
    st.markdown(""" 
    Yes! We've built a **Random Forest Classifier** that learns from features like email content, day sent, and user country to predict click likelihood.
    
    Instead of sending emails randomly, we can use the predicted probabilities to prioritize sending to users with high engagement potential.
    """)

    st.subheader("📌 By how much would the model improve CTR?")
    top_30 = email_df.sort_values(by='predicted_click_prob', ascending=False).head(int(len(email_df) * 0.3))
    optimized_ctr = top_30['clicked'].mean() * 100
    baseline_ctr = email_df['clicked'].mean() * 100
    improvement = optimized_ctr - baseline_ctr

    st.write(f"- **Baseline CTR:** {baseline_ctr:.2f}%")
    st.write(f"- **CTR with model (Top 30%):** {optimized_ctr:.2f}%")
    st.write(f"- **Improvement:** {improvement:.2f}%")

    st.markdown("We would test this by conducting an A/B test: group A gets emails using the model, group B receives randomly. Compare CTRs between both groups.")

    st.subheader("📌 Did you find interesting patterns in email campaign performance?")
    st.markdown("""
    Yes, here are some observations:
    
    - Certain **weekdays** have higher click-through rates.
    - Some **email versions** outperform others.
    - Users from specific **countries** show different engagement behaviors.

    These insights can be used to tailor campaigns for specific segments.
    """)

    def plot_segment(data, feature):
        plt.figure(figsize=(8, 4))
        sns.barplot(x=feature, y='clicked', data=data, estimator=np.mean)
        plt.title(f'Click Rate by {feature.capitalize()}')
        st.pyplot(plt)

    plot_segment(email_df, 'weekday')
    plot_segment(email_df, 'email_version')
    plot_segment(email_df, 'user_country')
