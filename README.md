## 🧓 Age Group Predictor - AIPlanet Hackathon Project (IIT Guwahati 2025)

This project was built as part of the **Summer Analytics 2025 Hackathon hosted by IIT Guwahati** on AIPlanet.
It predicts whether a person is an **Adult** or **Senior** based on their nutrition and health survey data.

### 🚀 Key Features
- Built a robust ML model using **XGBoost** + **Ensemble techniques**.
- Achieved high leaderboard score by applying **feature engineering**, **data cleaning**, and **tuning**.
- Deployed the model using a **Streamlit web app**.

---

## 📁 Project Structure
```
age-prediction-project/
├── data/                   # Processed datasets
├── notebooks/              # EDA and model building notebooks
├── models/                 # Saved model (.pkl)
├── streamlit_app/          # Streamlit UI
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```

---

## 🧠 Input Features
- `GLU`: Blood Glucose Level
- `INS`: Insulin Level
- `BMI`: Body Mass Index
- `PAQ605`: Physical Activity Level
- `PAD680`: Sleep Quality
- `DRQSPREP`: Dietary Preparedness

---

## ✅ How to Run
```bash
# Clone repo
https://github.com/yourusername/age-prediction-project.git
cd age-prediction-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run the app
cd streamlit_app
streamlit run app.py
```

---

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit for deployment

---

## 📌 Author
- 👤 [Your Name]
- 📧 your.email@example.com
- 🔗 [LinkedIn](https://linkedin.com/in/your-profile)
- 🔗 [GitHub](https://github.com/yourusername)


# 📦 requirements.txt
pandas
numpy
scikit-learn
xgboost
joblib
streaml
