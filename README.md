# Customer Churn Prediction (Machine Learning)

## 📌 Project Overview
Customer churn refers to when customers stop using a company’s product or service.  
In this project, we build a **machine learning model** to predict whether a customer is likely to churn based on their demographic, service, and billing information.

This project demonstrates a **complete end-to-end data science workflow**, from data understanding to model training and saving the trained model for future use.

---

## 🎯 Objective
- Understand customer churn data
- Preprocess and clean the dataset
- Train a machine learning model to predict churn
- Evaluate model performance using standard metrics
- Save the trained model and scaler for reuse

---

## 🗂️ Project Structure


---

## 🧪 Dataset Description
The dataset contains customer-level information such as:
- Demographics (gender, senior citizen)
- Service usage (internet service, contract type)
- Billing information (monthly charges, total charges)
- Target variable: **Churn** (Yes / No)

---

## ⚙️ Workflow

### 1️⃣ Data Understanding
- Loaded and explored the dataset
- Checked data types and missing values
- Identified the target variable (`Churn`)

### 2️⃣ Data Preprocessing
- Converted categorical variables using one-hot encoding
- Mapped target variable (`Yes` → 1, `No` → 0)
- Handled missing values
- Scaled numerical features using `StandardScaler`

### 3️⃣ Model Training
- Used **Logistic Regression** as a baseline model
- Performed train–test split with stratification
- Trained the model on scaled features

### 4️⃣ Model Evaluation
- Accuracy
- Classification Report (Precision, Recall, F1-score)
- ROC-AUC score

### 5️⃣ Model Saving
- Saved trained model using `joblib`
- Saved scaler for future predictions

---

## 📊 Results
The Logistic Regression model achieved:
- Good accuracy on unseen test data
- Balanced precision and recall
- Strong ROC-AUC score for churn prediction

(Exact metrics can be found in Notebook 3.)

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Joblib

---

## 🚀 How to Run the Project
1. Clone the repository
2. Install dependencies:
3. Run notebooks in order:
- `01_data_understanding.ipynb`
- `02_model_building.ipynb`
- `03_evaluation_and_saving.ipynb`

---

## 📌 Future Improvements
- Add advanced models such as SVM or Random Forest
- Perform hyperparameter tuning
- Add visualizations for insights
- Deploy the model as a web application

---

## 👤 Author
**Dona**  
Aspiring Data Scientist
GitHub: https://github.com/dopymol