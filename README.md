# Email_ML
Here's a detailed `README.md` file for your Email Campaign Click-Through Prediction App built using Streamlit and Random Forest:

---

📩 Email Campaign Click-Through Prediction App

This Streamlit web application helps marketers analyze email campaign performance and **predict click-through rates (CTR)** using a **Random Forest Classifier**. It provides insights into user engagement, optimizes email sending strategies, and visualizes campaign effectiveness.

🚀 Features

- Login & Signup System 
  Basic credential-based access with session management.

- Data Integration 
  Merges email send, open, and click logs into one comprehensive dataset.

- Model Training & Evaluation
  - Preprocesses data with label encoding.
  - Trains a Random Forest Classifier to predict email link clicks.
  - Displays accuracy, classification report, and confusion matrix.

- CTR Analysis  
  - Calculates baseline and optimized CTR (Top 30% targeting).
  - Visualizes CTR performance by weekday, email version, and country.

- Campaign Insights
  - Summarizes open and click rates.
  - Explains how the model improves CTR.
  - Displays meaningful trends and segment-level insights.

🧠 How the Model Works

- Input Features:
  - Email Text Type
  - Email Version
  - Weekday of Send
  - User Country

- Target: Whether the user clicked the email link (1 or 0).

- Model: Random Forest Classifier

- Evaluation: Accuracy score, confusion matrix, and classification report.

📁 Folder Structure

```
📦 EmailCampaignCTRApp
├── 📄 email_table.csv
├── 📄 email_opened_table.csv
├── 📄 link_clicked_table.csv
├── 📄 click_prediction_model.pkl
├── 📄 app.py
└── 📄 README.md
```

🛠️ Installation & Setup

1. Clone the Repository
   ```
   git clone https://github.com/your-username/EmailCampaignCTRApp.git
   cd EmailCampaignCTRApp
   ```
   
2. Install Required Packages 
   ```
   pip install -r requirements.txt
   ```

   > Or manually install:
   ```
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib
   ```
   
3. Run the App
   ```
   streamlit run app.py
   ```
   
📊 Sample Dashboard Preview

- Home: Campaign stats & intro
- Model Evaluation: Accuracy, report & confusion matrix
- CTR Analysis: Predictive targeting and lift in performance
- Campaign Insights: Strategies, user segmentation & observed patterns

📌 Future Improvements

- Integrate real-time campaign data from external APIs.
- Use advanced ML models like XGBoost or Neural Networks.
- Store users and credentials securely in a database.
- Enable downloadable performance reports and charts.

🤝 Contributions

Feel free to fork this project, raise issues, or submit pull requests. Let’s optimize email marketing together!
