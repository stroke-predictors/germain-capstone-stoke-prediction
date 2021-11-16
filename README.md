# Stroke Prediction

### Table of Contents
---

I.   [Project Overview             ](#i-project-overview)
1.   [Description                  ](#1-description)
2.   [Deliverables                 ](#2-deliverables)

II.  [Executive Summary             ](#ii-executive-summary)
1.   [Goals:                        ](#1-goals)
2.   [Key Findings:                 ](#2-key-findings)
3.   [Recommendations:              ](#1-recommendations)

III. [Project                      ](#iii-project)
1.   [Questions & Hypothesis       ](#1-questions--hypothesis)
2.   [Findings                     ](#2-findings)

IV. [Data Context                 ](#iv-data-context)
1.   [Data Dictionary              ](#1-data-dictionary)

V.  [Process                      ](#v-process)
1.   [Project Planning             ](#1-project-planning)
2.   [Data Acquisition             ](#2-data-acquisition)
3.   [Data Preparation             ](#3-data-preparation)
4.   [Data Exploration             ](#4-data-exploration)
5.   [Modeling & Evaluation        ](#5-modeling--evaluation)
6.   [Product Delivery             ](#6-product-delivery)

VI.   [Modules                      ](#vi-modules)

VII.  [Project Reproduction         ](#vii-project-reproduction)

<br>

<br>

### I. Project Overview
---

#### 1. Description


Identify stroke factors, using the stroke prediction dataset from Kaggle, build a predictive model that performs better than a baseline classification prediction.
You know, every 40 seconds someone in the United States has a stroke. And, one in every 6 cardiovascular disease deaths comes from stroke. Our team wants to reduce these numbers. We’re analyzing health data to identify the factors that are closely linked to risk of stroke. We’re also building models that predict whether a person is at risk of stroke.

#### 2. Deliverables

- Slides for project presentation.
- Final Report Notebook detailing all of the team's findings and methodologies.
- Sections indicated with markdown headings in the team's final notebook with a good title and the documentation is sufficiently explanatory and of high quality.
- A Python module or modules that automate the data acquisistion and preparation process, imported and used in final notebook.
- README file that details the project specs, planning, key findings, and steps to reproduce.
- Presentation by the team on the findings that is under 10 minutes.

### II. Executive Summary
---

#### 1. Goals:

- Incorporate classification methodologies and discover potential drivers of stroke using eleven clinical features for predicting stroke events.
- Create modules storing functions of each step of the data pipeline
- Thoroughly document each step
- Construct models
- Make sure project is reproduceable

#### 2. Key findings:
- 

#### 3. Recommendations:
- I

---

### III. Project

#### 1. Questions & Hypothesis

- What age group is most at risk?
- What drivers affect stroke?
- Are people over 65 significantly more likely to have a stroke?
- Are men more likely to have strokes?
- What pre-existing conditions have a correlation to a stroke?  (hypertension, heart_disease)

## Initial Hypotheses
### Hypothesis 1: Correlation Test (Stroke vs Age Group)
- $Ho$ : There is no relationship between age group and risk of stroke.
- $Ha$ : There is a relationship between age group and risk of stroke.

### Hypothesis 2: Correlation Test (Stroke vs < 65)
- $Ho$ : People over 65 years old are not more likely to have a stroke than people under 65.
- $Ha$ : People over 65 years old are more likely to have a stroke than people under 65.

### Hypothesis 3: Correlation Test (Stroke vs Don't Smoke)
- $Ho$ : People that currently smoke do not have a higher risk of stroke than people that don’t currently smoke.
- $Ha$ : People that currently smoke have a higher risk of stroke than people that don’t currently smoke.

### Hypothesis 4: Correlation Test (Stroke vs Men)
- $Ho$ : Men are not more at risk for stroke than women.
- $Ha$ : Men are more at risk for stroke than women.

### 2. Findings
#### My findings are:
- 


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
- [] Remedy any inconsistencies, duplicates, or structural problems within data
- [] Perform data summation

#### 3. Data Preparation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ 🟢 **Prepare** ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [] Address missing or inappropriate values, including outliers
- [] Plot distributions of variables
- [] Consider and create new features as needed
- [] Split data into `train`, `validate`, and `test`

#### 4. Data Exploration
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ 🟢 **Explore** ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [] Visualize relationships of variables
- [] Formulate hypotheses
- [] Perform statistical tests
- [] Decide upon features and models to be used

#### 5. Modeling & Evaluation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ 🟢 **Model** ➜ ☐ _Deliver_

- [] Establish baseline prediction
- [] Create, fit, and predict with models
    - Create at least four different models
    - Use different configurations of algorithms, hyper parameters, and/or features
- [] Evaluate models with out-of-sample data
- [] Utilize best performing model on `test` data
- [] Summarize, visualize, and interpret findings

#### 6. Product Delivery
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ ✓ _Model_ ➜ 🟢 **Deliver**
- [] Prepare Jupyter Notebook of project details through data science pipeline
    - Python code clearly commented when necessary
    - Sufficiently utilize markdown
    - Appropriately title notebook and sections
- [] With additional time, continue with exploration beyond MVP
- [] Proof read and complete README and project repository

### VI. Modules
---

The created modules used in this project below contain full comments an docstrings to better understand their operation.

### VII. Project Reproduction
---

To recreate and reproduce results of this project, you will need to

[[Return to Top]](#stroke-prediction)

