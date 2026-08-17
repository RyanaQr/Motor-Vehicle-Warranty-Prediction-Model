# 🚗 Motor-Vehicle Warranty Claim Prediction Using Machine Learning
Predictive Maintenance model for Motor warranty

## 📌 Project Overview

This project demonstrates an end-to-end **Machine Learning classification workflow** for predicting potential warranty-related failures using equipment and operational data.

The project uses the **AI4I 2020 Predictive Maintenance Dataset** and applies exploratory data analysis, feature preparation, and a **Random Forest Classifier** to predict machine failures.

The predictive maintenance problem is presented from a **warranty analytics perspective**, where early identification of potential failures can help organizations reduce warranty costs, improve product quality, optimize maintenance strategies, and support proactive decision-making.

---

## 🎯 Business Objective

Manufacturers and service organizations can incur significant costs when products fail during their warranty period.

The objective of this project is to:

* 🔍 Identify patterns associated with equipment failure
* 📊 Explore relationships between operational variables and failures
* 🤖 Develop a machine learning model to predict failures
* 📈 Evaluate model performance using multiple classification metrics
* 🧠 Identify the most important predictive features
* 💡 Generate insights that could support proactive maintenance and warranty decision-making

### Business Question

> **Can machine learning identify equipment that is more likely to experience a failure so that preventive action can be taken earlier?**

---

## 🗂️ Dataset

### AI4I 2020 Predictive Maintenance Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**, which contains operational and machine-related attributes used to predict machine failure.

The dataset includes variables such as:

* ⚙️ Product Type
* 🌡️ Air Temperature
* 🌡️ Process Temperature
* 🔄 Rotational Speed
* 🔧 Torque
* 🛠️ Tool Wear
* ❌ Machine Failure

The target variable used in this project is:

```text
Machine failure
```

Where:

* `0` → No machine failure
* `1` → Machine failure

---

## 🧰 Technologies & Libraries

### Programming Language

🐍 **Python**

### Data Analysis

* 🐼 Pandas
* 🔢 NumPy

### Data Visualization

* 📊 Matplotlib
* 📈 Seaborn

### Machine Learning

* 🤖 Scikit-learn
* 🌲 Random Forest Classifier
* 🔤 Label Encoding
* 📏 StandardScaler
* 📊 Classification Metrics

### Development Environment

* ☁️ Google Colab
* 💻 Jupyter Notebook
* 🐙 GitHub

---

## 📁 Project Structure

```text
Warranty-Claim-Prediction/
│
├── 📄 ai4i 2020 dataset.csv
│
├── 🐍 warranty_claim_prediction.py
│
├── 📓 Warranty Claim Prediction.ipynb
│
└── 📖 README.md
```

### File Descriptions

| File                                 | Description                                                            |
| ------------------------------------ | ---------------------------------------------------------------------- |
| 📄 `ai4i 2020 dataset.csv`           | Input dataset used for analysis and machine learning                   |
| 🐍 `warranty_claim_prediction.py`    | Python implementation of the complete analysis and Random Forest model |
| 📓 `Warranty Claim Prediction.ipynb` | Interactive Google Colab/Jupyter Notebook containing the analysis      |
| 📖 `README.md`                       | Project documentation                                                  |

---

# 🔄 Project Workflow

```text
📥 Data Collection
       ↓
🧹 Data Cleaning
       ↓
🔍 Exploratory Data Analysis
       ↓
🎯 Feature Selection
       ↓
🔤 Categorical Encoding
       ↓
✂️ Train/Test Split
       ↓
🌲 Random Forest Model
       ↓
🔮 Predictions
       ↓
📊 Model Evaluation
       ↓
📈 ROC-AUC Analysis
       ↓
⭐ Feature Importance
       ↓
💡 Business Insights
```

---

# 🔍 1. Data Loading

The dataset is loaded using Pandas.

Initial dataset inspection includes:

* Dataset dimensions
* Data types
* Duplicate records
* Missing values
* Basic data inspection

---

# 🧹 2. Data Cleaning

The project checks for duplicate records and missing values.

Missing values are examined.
This helps ensure that the dataset is suitable for subsequent analysis and modeling.

---

# 📊 3. Exploratory Data Analysis

Several EDA techniques are performed to understand the dataset.

### Dataset Analysis

* 📐 Number of rows and columns
* 🔎 Missing-value analysis
* 🔁 Duplicate-value analysis
* 📊 Statistical distributions
* 📦 Outlier detection

### Visualizations

* 📊 Machine failure distribution
* 📈 Numerical feature distributions
* 📦 Boxplots
* 🔥 Correlation heatmap
* 🔗 Pairplot
* 📊 Categorical variable distributions
* ⚙️ Failure by product type
* 🔄 Rotational speed vs. machine failure

The notebook also examines the distribution of the target variable:

---

# 🎯 4. Feature Selection

The target variable is separated from the predictor variables.
Where:
* `X` = Independent variables/features
* `y` = Target variable

Categorical variables are encoded using `LabelEncoder`.

---

# ✂️ 5. Train/Test Split

The dataset is divided into training and testing datasets.

### Split

* 🏋️ 80% → Training data
* 🧪 20% → Testing data
A fixed `random_state=42` is used to make the experiment reproducible.

---

# 📏 6. Feature Scaling

StandardScaler is applied to create scaled training and testing datasets.

The Random Forest model itself is trained using the original feature values because tree-based models do not require feature scaling.

---

# 🌲 7. Random Forest Model

The primary machine learning algorithm used is **Random Forest Classification**.

### Model Configuration

| Parameter       |         Value |
| --------------- | ------------: |
| Algorithm       | Random Forest |
| Number of Trees |           100 |
| Maximum Depth   |          None |
| Random State    |            42 |

The model is trained.

---

# 🔮 8. Prediction

The trained model generates predictions on unseen test data.

Prediction probabilities are also generated:
These probabilities are used for ROC-AUC analysis.

---

# 📊 9. Model Evaluation

The model is evaluated using multiple metrics rather than relying only on accuracy.

### Evaluation Metrics

* 🎯 Accuracy
* 🎯 Precision
* 🔎 Recall
* ⚖️ F1-Score
* 🧮 Confusion Matrix
* 📈 ROC Curve
* 🏆 ROC-AUC

This is particularly important for failure prediction because the positive class can be much smaller than the negative class.

---

## 🎯 Accuracy
Measures the percentage of total predictions that were correct.

---
## 🎯 Precision
Measures how many observations predicted as failures were actually failures.

---
## 🔎 Recall
Measures how many actual failures were successfully identified by the model.

For a failure/warranty prediction application, **recall is particularly important** because missing an actual failure can potentially result in higher warranty or maintenance costs.

---
## ⚖️ F1-Score
The F1-score balances precision and recall.

---

# 🧮 10. Confusion Matrix
The confusion matrix provides a detailed view of correct and incorrect predictions.

It helps identify:

* ✅ True Positives
* ✅ True Negatives
* ❌ False Positives
* ❌ False Negatives

---

# 📈 11. ROC Curve & ROC-AUC
The project generates ROC curves for both training and testing data.
ROC-AUC measures the model's ability to distinguish between failure and non-failure cases.

A higher ROC-AUC generally indicates better classification discrimination.

---

# ⭐ 12. Feature Importance
Random Forest provides feature importance values that help identify which variables contribute most to the model's predictions.

This allows us to answer an important business question:

> **Which operational factors are most strongly associated with predicted failures?**

These insights can potentially support:
* 🔧 Preventive maintenance
* 🏭 Manufacturing quality improvement
* 📦 Parts planning
* 🚗 Warranty risk analysis
* ⚙️ Equipment monitoring

---

# 💼 Business Value

A predictive warranty/failure model can help organizations move from a reactive maintenance strategy to a more proactive approach.

### Potential Applications

🔧 **Preventive Maintenance**

Identify high-risk equipment before failure occurs.

💰 **Warranty Cost Reduction**

Prioritize inspections and interventions for high-risk products.

📦 **Parts Planning**

Anticipate potential parts requirements based on predicted failures.

🏭 **Quality Improvement**

Identify operational factors associated with increased failure rates.

📊 **Operational Decision Making**

Provide data-driven insights to engineering and business teams.

---

# 🚀 Future Improvements

The current project establishes the baseline Random Forest model. Future versions can expand the solution with:

* 🔬 Hyperparameter tuning using `GridSearchCV`
* 🔄 Cross-validation
* ⚖️ SMOTE for class imbalance
* 🚀 XGBoost
* 🌳 LightGBM
* 🐈 CatBoost
* 🧠 SHAP Explainable AI
* 📊 Power BI dashboard
* 🌐 Streamlit prediction application
* ☁️ Cloud deployment
* 📡 Model monitoring
* 🔄 Data drift detection

---

# 🧠 Skills Demonstrated

This project demonstrates practical experience with:

```text
Python
Pandas
NumPy
Data Cleaning
Exploratory Data Analysis
Statistical Analysis
Feature Engineering
Feature Selection
Categorical Encoding
Machine Learning
Random Forest
Classification
Model Evaluation
Precision
Recall
F1-Score
ROC-AUC
Confusion Matrix
Feature Importance
Data Visualization
Predictive Analytics
```

---

# 📌 Key Takeaway

This project demonstrates how machine learning can be used to analyze operational data and predict potential failures before they occur.

The overall objective is not simply to build a machine learning model, but to demonstrate an end-to-end **data science workflow**:

> **Data → Analysis → Machine Learning → Evaluation → Predictive Insights → Business Decision**

---

# 👤 Author

**Ryana Quadir**

🎓 M.S. Computer & Information Science
🏫 University of Michigan–Dearborn

🔗 LinkedIn:
https://www.linkedin.com/in/ryana-quadir-81a15179/

💻 GitHub:
https://github.com/RyanaQr

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub.
