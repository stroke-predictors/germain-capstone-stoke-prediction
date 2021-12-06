# Stroke Prediction

![This is an image](https://i.imgur.com/mgshv8S.jpg)

### Table of Contents
---

I.   [Project Overview             ](#i-project-overview)
1.   [Description                  ](#1-description)
2.   [Deliverables                 ](#2-deliverables)
3.   [Team Members                 ](#3-team-members)

II.  [Executive Summary             ](#ii-executive-summary)
1.   [Goals                         ](#1-goals)
2.   [Key Findings                  ](#2-key-findings)
3.   [Recommendations               ](#3-recommendations)
4.   [Our Next Steps                ](#4-our-next-steps)

III. [Our Data Product Stroke Risk Calculator](#iii-Our-Data-Product-Stroke-Risk-Calculator)
1.   [Overview                     ](#1-overview)
2.   [How it works                 ](#2-how-it-works?)
3.   [How to use it                ](#3-how-to-use-it)

IV.  [Project                      ](#iii-project)
1.   [Hypotheses                   ](#1-hypotheses)
2.   [Findings                     ](#2-findings)
3.   [Model Takeaways              ](#3-model-takeaways)

V.   [Data Context                 ](#iv-data-context)
1.   [Data Dictionary              ](#1-data-dictionary)

VI.  [Process                      ](#v-process)
1.   [Project Planning             ](#1-project-planning)
2.   [Data Acquisition             ](#2-data-acquisition)
3.   [Data Preparation             ](#3-data-preparation)
4.   [Data Exploration             ](#4-data-exploration)
5.   [Modeling & Evaluation        ](#5-modeling--evaluation)
6.   [Product Delivery             ](#6-product-delivery)

VII.   [Modules                      ](#vi-modules)

VIII.  [Project Reproduction         ](#vii-project-reproduction)

<br>

<br>

### I. Project Overview
---

#### 1. Description

Did you know, **every 40 seconds** someone in the United States has a stroke and **every 4 minutes**, someone dies of stroke? Our team wants to reduce these numbers.

This project was created by the Stroke Prediction team from Codeup's Germain data science cohort. The project analyzes a stroke prediction dataset in an attempt to find the drivers of stroke and build a model that predicts stroke outcomes. The analysis from this project is used in our Stroke Risk Calculator, which incorporates our best model and generates a risk percentage for a user's inputs.


#### 2. Deliverables

- Slides for project presentation.
- Final Report Notebook detailing all of the team's findings and methodologies.
- Sections indicated with markdown headings in the team's final notebook with a good title and the documentation is sufficiently explanatory and of high quality.
- A Python module or modules that automate the data acquisistion and preparation process, imported and used in final notebook.
- README file that details the project specs, planning, key findings, and steps to reproduce.
- Presentation by the team on the findings that is under 10 minutes.

#### 3. Team Members

- Carolyn Davis
- Elihezer Lopez
- Jacob Paxton
- Sarah Lawson Woods
- Randy French

### II. Executive Summary
---

#### 1. Goals

- 1. Determine the drivers of stroke risk
- 2. Create an accurate predictive model for stroke outcomes
- 3. Predict the probability of stroke
- 4. Deliver a production-ready stroke risk calculator

<!-- - Incorporate classification methodologies and discover potential drivers of stroke using eleven clinical features for predicting stroke events.
- Create modules storing functions of each step of the data pipeline
- Thoroughly document each step
- Construct models
- Make sure project is reproduceable -->

#### 2. Key findings

- **Drivers of stroke risk:** Old age; Hyperglycemia/high glucose levels; hypertension; heart disease; marriage under 55 years old: never having married over 55 years old
- **Non-Drivers:** BMI; Gender; Residence location; Smoking status; Employment type
- **Best model's performance:** Recall: 83%, Accuracy: 65%, ROC AUC: 85%

#### 3. Recommendations
- People who fall into the high-risk categories we've found should consult with their doctor to get screened
- People can use our risk calculator to check their risk of stroke

#### 4. Our Next Steps
- With additional time, we would collect more records to train our model and conduct further multivariate analysis.

___

### III. Our Data Product: Stroke Risk Calculator
![This is an image](https://i.imgur.com/7xpF89F.png)

#### 1. Overview

- Provide users with a score indicating stroke risk where higher numbers are higher risk
- Uses our best model to calculate the risk score


#### 2. How it works?
- Re-creates the best-performing model from the Stroke Prediction team's analysis and fits it on the data used in the analysis.
- Uses sklearn's predict_proba method to calculate the risk of stroke
- Finally it returns the calculated number that indicates a percentage of risk for stroke.

#### 3. How to use it

- Run the risk_calculator.py file in command line or terminal.
- Imput the answers to the questions. 
- Program returns a calculated risk percentage.

---

### III. Project

### 1. Hypotheses

<!-- - What age group is most at risk?
- What drivers affect stroke?
- Are people over 65 significantly more likely to have a stroke?
- Are men more likely to have strokes?
- What pre-existing conditions have a correlation to a stroke?  (hypertension, heart_disease) -->

### Initial Hypotheses
#### Hypothesis 1: On average, an increase in BMI corresponds with an increase in average glucose level.
- *H<sub>0</sub>*: An increase in BMI does not correspond with an increase in average glucose level.
- *H<sub>a</sub>*: An increase in BMI corresponds with an increase in average glucose level.

#### Hypothesis 2: On average, a person who has had a stroke is older than someone who has not.
- *H<sub>0</sub>*: A person who has had a stroke is not statistically-significantly older than someone who has not.
- *H<sub>a</sub>*: A person who has had a stroke is statistically-significantly older than someone who has not.

#### Hypothesis 3: Smoking has an impact on stroke occurrence.
- *H<sub>0</sub>*: Smoking does not have an impact on stroke occurrence.
- *H<sub>a</sub>*: Smoking has an impact on stroke occurrence.

#### Hypothesis 4: Men and women have different stroke occurrence rates.
- *H<sub>0</sub>*: Men and women do not have different stroke occurrence rates.
- *H<sub>a</sub>*: Men and women have different stroke occurrence rates.

### 2. Findings
#### Our findings are: Initial Hypotheses - Combined Results

- We are 95% confident that an increase in BMI corresponds with an increase in average glucose level.
- We are 95% confident that people who have had a stroke are older on average than people who have not.
- We can't say with 95% confidence that smoking has an impact on stroke occurrence. 
- We can't say with 95% confidence that men and women have different stroke occurrence rates.

### 3. Model Takeaways:

- Baseline (Always Guess Stroke) Performance: Accuracy 5%, Recall 100%, ROC AUC 50%
- Our Best Model (Gaussian Naive-Bayes) Performance: Accuracy 65%, Recall 83%, ROC AUC 85%

### IV. Data Context
---

#### 1. Data Dictionary

Following acquisition and preparation of the initial Kaggle dataset, the DataFrames used in this project contain the following variables. Contained values are defined along with their respective data types.

| Feature               | Datatype      | Description                                          |
|:----------------------|:--------------|:-----------------------------------------------------|
| id                    | int64         | Unique ID                                            |
| gender                | object        | Male/ Female                                         |
| age                   | float         | Applicant age                                        |
| hypertension          | int64         | 0- If no hypertension, 1- If hypertension indicated  |         
| heart_disease         | int           | 0- If no heart disease, 1- If heart disease indicated|
| ever_married          | object        | Yes/No                                               | 
| work_type             | object        | Government job/ Self-employed/ Private/ Children     |       
| residence_type        | object        | Rural/ Urban                                         |
| avg_glucose_level     | float         | Number indicating their glucose level                |
| bmi                   | float         | Number indicating bmi score                          |
| smoking_status        | object        | Formerly smoked, Never smoked, Smokes, Unknown       |       
| stroke                | object        | 0- If no stroke 1- If stroke indicated               |           
|                       |               |                                                      |

   

### V. Process
---
- See the team's Trello board [Stroke Prediction](https://trello.com/b/9KkRcl2I/stroke-prediction)

#### 1. Project Planning
✓ 🟢 **Plan** ➜ ☐ _Acquire_ ➜ ☐ _Prepare_ ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Build this README containing:
    - Project overview
    - Initial thoughts and hypotheses
    - Project summary
    - Instructions to reproduce
- [x] Plan stages of project and consider needs versus desires

#### 2. Data Acquisition
✓ _Plan_ ➜ 🟢 **Acquire** ➜ ☐ _Prepare_ ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Obtain initial data and understand its structure
    - Obtain data from Kaggle - https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
- [x] Remedy any inconsistencies, duplicates, or structural problems within data
- [x] Perform data summation

#### 3. Data Preparation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ 🟢 **Prepare** ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Address missing or inappropriate values, including outliers
- [x] Plot distributions of variables
- [x] Consider and create new features as needed
- [x] Split data into `train`, `validate`, and `test`

#### 4. Data Exploration
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ 🟢 **Explore** ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Visualize relationships of variables
- [x] Formulate hypotheses
- [x] Perform statistical tests
- [x] Decide upon features and models to be used

#### 5. Modeling & Evaluation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ 🟢 **Model** ➜ ☐ _Deliver_

- [x] Establish baseline prediction
- [x] Create, fit, and predict with models
    - Create at least four different models
    - Use different configurations of algorithms, hyper parameters, and/or features
- [x] Evaluate models with out-of-sample data
- [x] Utilize best performing model on `test` data
- [x] Summarize, visualize, and interpret findings

#### 6. Product Delivery
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ ✓ _Model_ ➜ 🟢 **Deliver**
- [x] Prepare Jupyter Notebook of project details through data science pipeline
    - Python code clearly commented when necessary
    - Sufficiently utilize markdown
    - Appropriately title notebook and sections
- [x] With additional time, continue with exploration beyond MVP
- [x] Proof read and complete README and project repository

### VI. Modules
---

The created modules used in this project below contain full comments and docstrings to better understand their operation.

- prepare.py
- model.py
- risk_calculator.py

### VII. Project Reproduction
---

To recreate and reproduce results of this project, you will need to:

- Read this README.md
- Download these files from the repository - healthcare-dataset-stroke-data.csv, prepare.py, model.py, risk_calculator.py and final_notebook.ipynb files into your working directory.
- To use the SMOTE+Tomek to eliminate class imbalances for train split, you will need to install the tool kit using pip or conda.
   - pip install -U imbalanced-learn
   - conda install -c conda-forge imbalanced-learn
- Run the final_notebook.ipynb notebook

** The dataset was downloaded from the website https://www.kaggle.com/fedesoriano/stroke-prediction-dataset on 11/15/2021. The orginial data set may have changed since we completed the project.**


[[Return to Top]](#stroke-prediction)
