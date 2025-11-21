# Titanic Survival Prediction

A machine learning classification project that predicts Titanic passenger survival with **82% accuracy** using Random Forest algorithm.

##  Project Overview

This project analyzes the famous Titanic dataset to predict passenger survival using machine learning classification algorithms. The sinking of the RMS Titanic on April 15, 1912, resulted in the deaths of 1,502 out of 2,224 passengers and crew. This analysis reveals the key factors that determined survival.

**Goal**: Build a predictive model to determine which passengers were more likely to survive based on demographic and socioeconomic features.

**Key Achievement**: Achieved **82% prediction accuracy** using Random Forest Classifier.

---

## Key Findings

### Top 3 Survival Factors

1. **Gender** (74% female vs 19% male survival rate)
   - Clear evidence of "women and children first" evacuation protocol
   
2. **Passenger Class** (63% 1st class vs 24% 3rd class survival)
   - Socioeconomic status significantly impacted access to lifeboats
   
3. **Age** (Children had higher survival rates)
   - Younger passengers were prioritized during evacuation

### Visualizations

![Survival by Gender](images/survival_by_gender.png)
*Female passengers had 4x higher survival rate than males*

![Survival by Class](images/survival_by_class.png)
*First-class passengers had significantly better survival odds*

![Correlation Heatmap](images/correlation_heatmap.png)
*Feature correlation analysis reveals key relationships*

![Feature Importance](images/feature_importance.png)
*Random Forest feature importance rankings*

---

## Dataset

**Source**: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)

**Size**: 891 passengers (training data)

**Features**:
| Feature | Description | Type |
|---------|-------------|------|
| PassengerId | Unique passenger identifier | Numeric |
| Survived | Survival status (0 = No, 1 = Yes) | Binary (Target) |
| Pclass | Ticket class (1st, 2nd, 3rd) | Ordinal |
| Name | Passenger name | Text |
| Sex | Gender | Categorical |
| Age | Age in years | Numeric |
| SibSp | Number of siblings/spouses aboard | Numeric |
| Parch | Number of parents/children aboard | Numeric |
| Ticket | Ticket number | Text |
| Fare | Passenger fare | Numeric |
| Cabin | Cabin number | Text |
| Embarked | Port of embarkation (C/Q/S) | Categorical |

**Target Variable**: Survived (0 = No, 1 = Yes)

**Data Quality Issues**:
| Column | Missing Values | Action Taken |
|--------|---------------|--------------|
| Age | 177 (19.9%) | Imputed with median |
| Cabin | 687 (77.1%) | Dropped (too sparse) |
| Embarked | 2 (0.2%) | Filled with mode (most common port) |

---

##  Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualizations
- **scikit-learn** - Machine learning models and evaluation
- **Jupyter Notebook** - Interactive development environment

---

##Modelallation & Usage

### Prerequisites


pip install -r requirements.txt
Requirements.txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
Run the Project
# Clone the repository
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook titanic_analysis.ipynb

 Methodology
1. Data Exploration
Analyzed distributions and survival rates across different features
Identified missing values and outliers
Discovered key patterns through visualizations
2. Data Preprocessing
Missing Value Handling:
Age: Filled with median (28 years) - robust to outliers
Embarked: Filled with mode (Southampton) - most common port
Cabin: Dropped due to 77% missing data
Feature Engineering:
FamilySize = SibSp + Parch + 1 (total family members aboard)
IsAlone = Binary indicator for solo travelers
Title = Extracted from name (Mr., Mrs., Miss., Master., Rare titles)
Encoding Categorical Variables:
Sex: Binary encoding (male=1, female=0)
Embarked: Label encoding (S=0, C=1, Q=2)
Title: One-hot encoding (creates separate binary columns for each title)
3. Feature Selection
Dropped irrelevant features: PassengerId, Name, Ticket, Cabin
Selected 15 engineered features for modeling
4. Model Development
Train-Test Split: 80-20 with stratification to maintain class balance
Feature Scaling: StandardScaler for Logistic Regression
Models Tested:
Logistic Regression (baseline)
Random Forest Classifier (final model)
5. Model Evaluation
Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
Validation: 5-fold cross-validation for robustness
Feature Importance Analysis: Identified key survival predictors

##  Model Development and Results

### Models Tested
1. **Logistic Regression** (Baseline)
2. **Random Forest Classifier** (Advanced)

### Model Performance

| Metric | Logistic Regression | Random Forest | Winner |
|--------|-------------------|---------------|---------|
| **Accuracy** | ~80% | ~82% | RF |
| **Precision** | 0.78 | 0.81 | RF |
| **Recall** | 0.73 | 0.76 | RF |
| **F1-Score** | 0.75 | 0.78 | RF |
| **Training Time** | < 1 sec | ~2 sec | LR |

### Confusion Matrix Analysis (Random Forest)
Predicted
            Dead  Alive
Actual  Dead    [95]  [15]   → 86% correctly identified deaths
Alive   [17]  [52]   → 75% correctly identified survivors
**Error Analysis:**
- **False Positives (15)**: Predicted survival but died
  - Likely edge cases (e.g., women in 3rd class with limited lifeboat access)
- **False Negatives (17)**: Predicted death but survived
  - Possibly men who found spots in lifeboats or crew members

---

## Feature Importance Rankings

### Top 5 Most Important Features (Random Forest)

| Rank | Feature | Importance | Interpretation |
|------|---------|-----------|----------------|
| 1 | Sex | 0.28 | Gender dominated survival predictions |
| 2 | Title_Mr | 0.19 | Being an adult male strongly predicted death |
| 3 | Fare | 0.12 | Wealth proxy, correlated with class |
| 4 | Age | 0.11 | Younger passengers had slight advantage |
| 5 | Pclass | 0.09 | Class still mattered after accounting for fare |

### Insights
- **Sex/Title features combined**: ~47% of model decisions
  - Confirms gender was the dominant factor
- **Economic indicators (Fare + Pclass)**: ~21% of model decisions
  - Wealth clearly provided survival advantage
- **Age**: 11% importance
  - Moderate effect, supporting "women and children" but less emphasis on children than expected

---

##  Model Validation and Robustness

### Cross-Validation Results
- **5-Fold CV Accuracy**: 81.5% ± 2.3%
- **Stability**: Low standard deviation indicates model is robust across different data splits

### Overfitting Check
- **Training Accuracy**: 85%
- **Test Accuracy**: 82%
- **Gap**: 3% (acceptable; no severe overfitting)

### Holdout Set Performance
- Final model achieved **82.1% accuracy** on completely unseen test data
- Confirms model generalizes well beyond training data

---

##  Business Insights and Recommendations

### Historical Insights
1. **Evacuation protocol was followed unevenly**
   - Women in 1st/2nd class: ~95% survival
   - Women in 3rd class: ~50% survival
   - Indicates structural barriers (locked gates, deck location) prevented equal access

2. **Wealth significantly impacted survival beyond protocol**
   - Even among men, 1st class had 3x survival rate of 3rd class
   - Suggests wealthier passengers had faster/better access to information and lifeboats

3. **Family dynamics mattered**
   - Small families (2-4 members) had best survival rates
   - Large families struggled (possibly trying to stay together)
   - Solo travelers had moderate survival (more mobile but no group support)

### Modern Applications

**For Emergency Response Planning:**
- Ensure evacuation routes are equally accessible across all socioeconomic areas
- Account for family units in evacuation plans (designated meeting points)
- Provide clear communication to all demographics simultaneously

**For Risk Assessment:**
- Demographics (age, gender) remain relevant in disaster modeling
- Location/accessibility is critical (3rd class was deep in ship)
- Social factors (family size) affect evacuation behavior

**For Future Data Collection:**
- Cabin location (which deck) would be valuable but was too sparse in this dataset
- Crew member identification could reveal different survival patterns
- Lifeboat assignments would enable direct causality analysis

---

##  Limitations and Future Work

### Current Limitations

**Data Sparsity**
   - 77% missing cabin data prevented location-based analysis
   - Would have been valuable to analyze deck location vs. survival

 **Correlation vs. Causation**
   - Model identifies correlations, not causal mechanisms
   - Example: Fare predicts survival, but fare doesn't *cause* survival—it's a proxy for class/location

 **Model Interpretability**
   - Random Forest is somewhat "black box"
   - Feature importance shows *what* matters but not *how* features interact

  **Limited External Validity**
   - Model trained on historical disaster with specific protocol
   - May not generalize to modern evacuation scenarios

### Future Improvements

**Data Enhancement:**
- Incorporate cabin location data (if available from other sources)
- Add crew member indicators
- Include lifeboat assignment information

**Advanced Modeling:**
- Try Gradient Boosting (XGBoost, LightGBM) for potentially better performance
- Implement SHAP values for better model explainability
- Build ensemble model combining multiple algorithms

**Feature Engineering:**
- Create interaction features (e.g., Sex × Pclass)
- Engineer deck level from cabin numbers
- Add time-related features if embarkation times were available

**Business Application:**
- Develop risk score calculator for individual passengers
- Build interactive dashboard for exploring "what-if" scenarios
- Create survivor probability heat map by passenger characteristics

Contact
[Lawal Sunmisola Barakat] Email:lawalsunmisola2020@gmail.com
