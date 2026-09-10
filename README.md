# 🚀 Tourism Experience Analytics: Classification, Prediction & Recommendation System

## 📌 Project Overview

**Tourism Experience Analytics** is an end-to-end Machine Learning project focused on analyzing tourism and visitor experience data.

The project uses historical tourism transaction, user, attraction, location, and rating data to:

- Understand visitor behavior
- Analyze tourism trends and patterns
- Predict visitor **Visit Mode**
- Predict **Tourism Experience Rating**
- Generate personalized **Attraction Recommendations**
- Build an interactive **Streamlit application**
- Track Machine Learning experiments using **MLflow**

The project combines **Data Analytics, Machine Learning, Recommendation Systems, and Streamlit** into a single tourism analytics solution.

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Perform data cleaning and preprocessing on tourism datasets.
2. Explore visitor behavior using Exploratory Data Analysis (EDA).
3. Identify important patterns in tourism transactions.
4. Build a classification model to predict visitor Visit Mode.
5. Build a regression model to predict attraction ratings.
6. Develop a personalized recommendation system.
7. Track and compare ML models using MLflow.
8. Deploy the final solution using Streamlit.
9. Generate meaningful business insights from tourism data.

---

## 🏗️ Project Architecture

```text
Raw Tourism Data
       │
       ▼
Data Cleaning & Preprocessing
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Feature Engineering
       │
       ├───────────────┐
       ▼               ▼
Classification      Regression
       │               │
       ▼               ▼
Visit Mode          Rating Prediction
       │               │
       └───────┬───────┘
               ▼
      Recommendation System
               │
               ▼
        MLflow Tracking
               │
               ▼
       Streamlit Application
               │
               ▼
        Tourism Insights
```

---

## 📂 Dataset

The project uses multiple related tourism datasets.

### Dataset Files

| Dataset | Description |
|---|---|
| Transaction.xlsx | Visitor transaction, visit mode, attraction and rating information |
| User.xlsx | User location and geographic information |
| City.xlsx | City information |
| Updated_Item.xlsx | Attraction information |
| Type.xlsx | Attraction type information |
| Mode.xlsx | Visitor visit mode information |
| Continent.xlsx | Continent information |
| Country.xlsx | Country information |
| Region.xlsx | Region information |

### Dataset Dimensions

| Dataset | Rows | Columns |
|---|---|---|
| Transaction | 52,930 | 7 |
| User | 33,530 | 5 |
| City | 9,143 | 3 |
| Updated Item | 1,698 | 5 |
| Type | 17 | 2 |
| Mode | 6 | 2 |
| Continent | 6 | 2 |
| Country | 165 | 3 |
| Region | 22 | 3 |

---

## 🔍 Key Features

The project works with information such as:

- User ID
- Transaction ID
- Visit Year
- Visit Month
- Visit Mode
- Attraction ID
- Rating
- Continent
- Region
- Country
- City
- Attraction Type
- Attraction Name
- Attraction Address

---

## 🧹 Data Cleaning & Preprocessing

The following preprocessing steps were performed:

- Checked dataset dimensions
- Inspected data types
- Checked missing values
- Checked duplicate records
- Identified placeholder values
- Handled missing geographic information
- Validated rating ranges
- Validated visit month values
- Validated visit mode values
- Removed duplicate transaction records where required
- Joined related lookup tables
- Created a consolidated tourism master dataset

The final processed dataset is saved as:

```
outputs/cleaned_tourism_master.csv
```

---

## 📊 Exploratory Data Analysis

EDA was performed to understand tourism behavior and visitor preferences.

### Analysis Areas

- Visitor distribution by Visit Mode
- Rating distribution
- Monthly tourism trends
- Yearly tourism trends
- Attraction popularity
- Attraction ratings
- Country-wise visitors
- Region-wise visitors
- Continent-wise visitors
- City-level tourism patterns
- Visitor preferences
- Attraction type analysis

### Business Insights

Some important observations include:

- Ratings are concentrated mainly around higher values such as 4 and 5.
- Couples and Family visitors represent major Visit Mode groups.
- Business visitors represent a comparatively small group.
- Geographic analysis helps identify important tourism source markets.
- Attraction popularity can be used to support personalized recommendations.

---

## 🤖 Machine Learning

### 1️⃣ Classification — Visit Mode Prediction

**Objective**

Predict the visitor's Visit Mode based on tourism and visitor-related features.

**Target Variable**

`VisitMode`

**Visit Modes**

- Business
- Couples
- Family
- Friends
- Solo

**Models Evaluated**

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Other classification approaches used during experimentation

**Evaluation Metrics**

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

**Best Model**

Random Forest Classifier

The final classification model is saved as:

```
outputs/classification_model.pkl
```

**Classification Performance**

The final model achieved approximately:

```
Accuracy   : 0.49
Macro F1   : 0.29
Weighted F1: 0.44
```

The relatively low macro F1 highlights the effect of class imbalance, particularly for the Business category.

---

### 📈 2️⃣ Regression — Rating Prediction

**Objective**

Predict the tourism experience Rating using visitor, location, attraction, and visit-related features.

**Target Variable**

`Rating`

**Rating Range**

1 - 5

**Models Evaluated**

- Linear Regression
- Random Forest Regressor
- Gradient Boosting
- LightGBM
- Other regression models used during experimentation

**Evaluation Metrics**

- MAE
- MSE
- RMSE
- R² Score

**Best Model**

LightGBM

The final regression model is saved as:

```
outputs/regression_model.pkl
```

The model evaluation showed that rating prediction is challenging because historical ratings are concentrated in a narrow range and may not be strongly explained by the available features.

---

### 🎯 3️⃣ Recommendation System

A personalized tourism recommendation system was developed to recommend attractions based on visitor preferences.

The recommendation engine combines:

**🔹 Collaborative Filtering**

Uses user-attraction rating behavior to identify attractions preferred by similar users.

**🔹 Content-Based Filtering**

Uses attraction characteristics such as:

- Attraction Type
- Average Rating
- Popularity

**🔹 Hybrid Recommendation**

The final recommendation approach combines:

```
Collaborative Filtering
          +
Content-Based Filtering
          =
Hybrid Recommendation System
```

The hybrid system can provide personalized attraction suggestions while avoiding attractions that the user has already visited.

---

## 💾 Model & Recommendation Artifacts

The following trained artifacts are included in the `outputs/` directory:

```
outputs/
│
├── cleaned_tourism_master.csv
├── classification_model.pkl
├── regression_model.pkl
├── label_encoders.pkl
├── user_item_matrix.pkl
├── item_similarity.pkl
├── content_similarity.pkl
└── attraction_features.pkl
```

The `.pkl` model files are managed using Git LFS because of their size.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit application.

The application provides functionality for:

### 📊 Dashboard

- Tourism KPIs
- Visitor statistics
- Rating analysis
- Visit Mode distribution
- Geographic insights
- Attraction insights

### 🔮 Prediction

Users can provide tourism-related inputs to obtain:

- Predicted Visit Mode
- Predicted Rating

### 🎯 Recommendation

Users can enter a User ID and receive personalized attraction recommendations.

---

## 🚀 Run the Streamlit Application Locally

### Step 1 — Clone the Repository

```bash
git clone https://github.com/roshan2156/Tourism-Experience-Analytics.git
```

### Step 2 — Navigate to the Project

```bash
cd Tourism-Experience-Analytics
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run Streamlit

```bash
streamlit run app.py
```

The application will open in your browser.

Usually:

```
http://localhost:8501
```

---

## 📁 Project Structure

```
Tourism-Experience-Analytics/
│
├── data/
│       ├── Transaction.xlsx
│       ├── User.xlsx
│       ├── City.xlsx
│       ├── Updated_Item.xlsx
│       ├── Type.xlsx
│       ├── Mode.xlsx
│       ├── Continent.xlsx
│       ├── Country.xlsx
│       └── Region.xlsx
│
├── outputs/
│   ├── cleaned_tourism_master.csv
│   ├── classification_model.pkl
│   ├── regression_model.pkl
│   ├── label_encoders.pkl
│   ├── user_item_matrix.pkl
│   ├── item_similarity.pkl
│   ├── content_similarity.pkl
│   └── attraction_features.pkl
│
├── app.py
├── requirements.txt
├── Tourism_Experience_Analytics (1).ipynb
├── .gitattributes
└── README.md
```

---

## 🛠️ Technologies Used

**Programming**
- Python

**Data Processing**
- Pandas
- NumPy

**Data Visualization**
- Matplotlib
- Seaborn

**Machine Learning**
- Scikit-learn
- LightGBM

**Recommendation System**
- Cosine Similarity
- Collaborative Filtering
- Content-Based Filtering
- Hybrid Recommendation

**Deployment**
- Streamlit

**Experiment Tracking**
- MLflow

**Version Control**
- Git
- GitHub
- Git LFS

---

## 📌 Skills Demonstrated

This project demonstrates practical experience in:

- Data Cleaning
- Data Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- Statistical Analysis
- Classification
- Regression
- Recommendation Systems
- Model Evaluation
- MLflow
- Streamlit
- Git & GitHub
- Git LFS
- Business Intelligence
- Data-driven Decision Making

---

## 💼 Business Use Cases

The Tourism Experience Analytics system can help tourism businesses and platforms:

**🏨 Tourism Businesses**

Understand visitor behavior and preferences.

**🗺️ Travel Platforms**

Recommend attractions based on user interests.

**📍 Destination Management**

Identify popular attractions and tourism patterns.

**👥 Customer Segmentation**

Understand different visitor groups such as:

- Couples
- Families
- Friends
- Solo travelers
- Business travelers

**⭐ Experience Analysis**

Analyze attraction ratings and visitor satisfaction patterns.

---

## 🔮 Future Improvements

Future versions of the project can include:

- Deep Learning recommendation models
- Advanced NLP for tourism reviews
- Real-time recommendation engine
- More advanced user segmentation
- Hyperparameter optimization
- Explainable AI
- Real-time tourism APIs
- Weather-aware recommendations
- Budget-based attraction recommendations
- Location-aware recommendations
- Advanced collaborative filtering
- A/B testing of recommendation strategies

---

## 📊 Project Results

| Component | Best Approach |
|---|---|
| Visit Mode Prediction | Random Forest |
| Rating Prediction | LightGBM |
| Recommendation | Hybrid Collaborative + Content-Based |
| Deployment | Streamlit |
| Experiment Tracking | MLflow |
| Version Control | Git + GitHub + Git LFS |

---

## 🌐 Project Links

**💻 GitHub Repository**

[Github Repository](https://github.com/roshan2156/Tourism-Experience-Analytics.git)

**🌐 Live Streamlit Application**

[Streamlit live Application](https://tourism-experience-analytics-md9xrgqr9uvappukrrappo3w.streamlit.app/)

**📓 Notebook**

[Google Colab Notebook](https://colab.research.google.com/drive/1UHKjQ6bW2R6bX0OnF8vY_6hdZVyuTzXN?usp=sharing)

---

## 👨‍💻 Author

**Roshan Patil**

---

## 🎓 Training

This project was completed as part of my AI/ML training/project work at Labmentix.

---

## ⭐ If you find this project useful

Feel free to ⭐ star the repository and explore the project!
