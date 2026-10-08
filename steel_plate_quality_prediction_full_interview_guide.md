# Steel Plate Quality Prediction --- Full Project Interview Guide

## 1. Project in one sentence

An AI/ML-based web application that takes **27 numerical measurements
describing a steel-plate defect** and predicts which of **7 fault
categories** the defect most likely belongs to, while also showing class
probabilities and SHAP-based explanations.

## 2. What problem does it solve?

Imagine a steel factory. A steel plate develops a surface defect. An
inspection process produces numerical measurements describing that
defect.

Instead of manually deciding the fault category, the trained ML model
analyzes the measurements and predicts the most likely fault.

``` text
Steel plate
    ↓
Surface defect
    ↓
27 measurements
    ↓
Machine-learning model
    ↓
Fault category
    ↓
Probability + explanation
```

This is a **software / AI-ML project**, not a hardware project. The
current application does not directly operate sensors, cameras, factory
machines, or embedded hardware.

## 3. What does the project predict?

The current project does **not** simply predict `Good` vs `Bad`.

It predicts one of seven fault categories:

1.  Pastry
2.  Z_Scratch
3.  K_Scatch
4.  Stains
5.  Dirtiness
6.  Bumps
7.  Other_Faults

Therefore, the problem is **multiclass classification**.

## 4. Dataset

The project uses the **Steel Plates Faults** dataset from the UCI
Machine Learning Repository.

The dataset contains approximately:

-   1,941 records
-   27 input features
-   7 fault categories

Think of one historical record as:

``` text
27 measurements + known fault category
```

For example:

``` text
Record 1 → measurements → Bumps
Record 2 → measurements → Stains
Record 3 → measurements → Pastry
...
```

The model learns patterns from these labeled historical examples.

## 5. What does one row represent?

One row represents an observed steel-plate defect described by numerical
measurements and a known fault category.

The measurements describe characteristics such as:

-   Position
-   Size
-   Perimeter
-   Luminosity
-   Shape
-   Edges
-   Orientation
-   Steel type
-   Plate thickness
-   Other transformed measurements

## 6. The 27 input features

The deployed application expects these 27 feature columns.

### Position

-   `X_Minimum` --- minimum X-coordinate of the defect region
-   `X_Maximum` --- maximum X-coordinate
-   `Y_Minimum` --- minimum Y-coordinate
-   `Y_Maximum` --- maximum Y-coordinate

### Size / geometry

-   `Pixels_Areas` --- defect area in pixels
-   `X_Perimeter` --- X-related perimeter measurement
-   `Y_Perimeter` --- Y-related perimeter measurement

### Luminosity

-   `Sum_of_Luminosity`
-   `Minimum_of_Luminosity`
-   `Maximum_of_Luminosity`
-   `Luminosity_Index`

These describe brightness/luminosity characteristics.

### Manufacturing / steel information

-   `Length_of_Conveyer`
-   `TypeOfSteel_A300`
-   `TypeOfSteel_A400`
-   `Steel_Plate_Thickness`

The steel-type columns are categorical information represented
numerically.

### Shape / edge characteristics

-   `Edges_Index`
-   `Empty_Index`
-   `Square_Index`
-   `Outside_X_Index`
-   `Edges_X_Index`
-   `Edges_Y_Index`
-   `Outside_Global_Index`

### Transformed features

-   `LogOfAreas`
-   `Log_X_Index`
-   `Log_Y_Index`

### Other characteristics

-   `Orientation_Index`
-   `SigmoidOfAreas`

## 7. Input and output

The basic ML representation is:

``` text
INPUT (X)
27 measurements
       ↓
Machine-learning model
       ↓
OUTPUT (Y)
1 of 7 fault categories
```

`X` means features/inputs.

`Y` means target/label/output.

The model learns an approximation of:

``` text
f(X) → Y
```

In simple language:

> Given the 27 measurements, predict the fault category.

## 8. Why is it classification?

Because the output is categorical:

``` text
Bumps
Stains
Pastry
Z_Scratch
K_Scatch
Dirtiness
Other_Faults
```

It is not a continuous numerical prediction.

Because there are seven possible categories, it is **multiclass
classification**.

## 9. Training vs prediction

### Training

Historical labeled data is used to learn patterns:

``` text
Historical data
     ↓
Features + known labels
     ↓
ML algorithm
     ↓
Learned patterns
     ↓
Trained model
```

### Prediction / inference

After training:

``` text
New 27 measurements
     ↓
Already-trained model
     ↓
Prediction
```

Your Streamlit application is primarily an **inference application**.

## 10. Does the model predict the original 1,941 records?

Not automatically.

The 1,941 records are historical examples used for training/evaluation.

After training, the model can predict new unseen records.

``` text
1,941 historical records
        ↓
Training
        ↓
Trained model
        ↓
New 27 values
        ↓
Prediction
```

The number of predictions is not limited to 1,941.

The current CSV interface, however, intentionally uses only the **first
row** of an uploaded CSV.

## 11. Can you give new 27 inputs that were not in the dataset?

Yes.

The exact combination of 27 values does not have to appear in the
training dataset.

That is the purpose of **generalization**.

``` text
Training examples
       ↓
Learn patterns
       ↓
New unseen 27-value input
       ↓
Prediction
```

The new data should still have the same feature meanings, compatible
units/schema, valid values, and preferably a realistic distribution.

## 12. Is the dataset permanently fixed?

The current trained model has a **fixed expected input schema**.

However, the training dataset can be expanded later with compatible
steel-plate data.

``` text
Original training data
       +
New labeled steel data
       ↓
Retraining
       ↓
New model
```

A completely unrelated dataset cannot simply be given to this model.

## 13. Machine-learning model

The deployed application uses a **tree-based classification approach**,
and its process-monitoring logic refers to Random Forest feature
importance.

You should understand Random Forest thoroughly.

## 14. Decision Tree

A decision tree repeatedly asks feature-based questions.

Simplified example:

``` text
Is Pixels_Areas > X?
       /            Yes        No
      ↓          ↓
Is Shape > Y?   ...
    /      Yes     No
   ↓       ↓
 Bumps   Stains
```

Each question is a split. Eventually the tree reaches a leaf and
predicts a class.

## 15. Random Forest

Random Forest combines many decision trees.

``` text
                 Input
                   │
       ┌───────────┼───────────┐
       ↓           ↓           ↓
     Tree 1      Tree 2      Tree 3
       ↓           ↓           ↓
     Bumps       Bumps       Stains
       │           │           │
       └───────────┼───────────┘
                   ↓
            Final prediction
                   ↓
                 Bumps
```

The ensemble is generally more robust than relying on a single tree.

## 16. Why Random Forest?

It is a suitable choice for structured/tabular data because it can:

-   Capture nonlinear relationships
-   Capture feature interactions
-   Be relatively robust
-   Work well with numerical/tabular data
-   Provide feature importance
-   Avoid the same feature-scaling requirements as many distance-based
    algorithms

Do not claim exact hyperparameters unless you recover the original
training code.

## 17. What is bagging?

Bagging means **Bootstrap Aggregating**.

Different trees can be trained on different bootstrap samples of the
training data and their predictions can be combined.

``` text
Original data
    │
    ├── Bootstrap sample → Tree 1
    ├── Bootstrap sample → Tree 2
    ├── Bootstrap sample → Tree 3
    └── ...
```

## 18. Gini impurity

A decision tree can use Gini impurity to measure how mixed the classes
are.

``` text
Gini = 1 - Σ(p_i²)
```

where `p_i` is the proportion of class `i` in the node.

A completely pure node has Gini impurity 0.

## 19. Entropy

Another tree-splitting measure is entropy:

``` text
Entropy = -Σ p_i log2(p_i)
```

A pure node has entropy 0.

## 20. Feature scaling

Tree-based models generally do not require feature scaling in the same
way as distance-based methods such as KNN.

Trees make threshold-based decisions such as:

``` text
Pixels_Areas > 500?
```

rather than relying on feature distances.

Do not claim a specific preprocessing step unless it is verified in your
original training code.

## 21. Overfitting

Overfitting occurs when a model learns training data too specifically.

Example:

``` text
Training performance = very high
Test performance     = much lower
```

The goal is good performance on unseen data.

## 22. Underfitting

Underfitting occurs when the model is too simple to learn useful
patterns.

``` text
Training performance = poor
Test performance     = poor
```

## 23. Generalization

Generalization is the ability to make useful predictions on unseen data.

That is why your model can receive a new 27-value combination.

## 24. Model evaluation

Important metrics for this multiclass problem include:

### Accuracy

``` text
Correct predictions
-------------------
Total predictions
```

### Precision

Of the examples predicted as a class, how many were actually that class?

### Recall

Of the actual examples belonging to a class, how many were detected?

### F1-score

``` text
F1 = 2 × Precision × Recall
     -----------------------
      Precision + Recall
```

### Confusion matrix

A matrix comparing actual classes with predicted classes.

Diagonal = correct predictions.

Off-diagonal = misclassifications.

### Important warning

The current GitHub repository contains the serialized model and
deployment code but not the original training notebook/experiment log.
Therefore, do **not invent exact accuracy, precision, recall, F1 or
confusion-matrix values**.

## 25. Model serialization

The repository contains:

``` text
steel_fault_model_bundle.pkl
```

The trained model is saved so that the deployed application does not
need to retrain every time.

Conceptually:

``` text
Training
   ↓
Trained model
   ↓
Serialize with Joblib
   ↓
.pkl file
```

Later:

``` text
.pkl file
   ↓
joblib.load()
   ↓
Model in memory
```

## 26. What is inside the model bundle?

The application expects:

``` python
bundle["model"]
bundle["feature_columns"]
bundle["target_classes"]
```

Therefore the bundle contains the trained model and metadata needed for
inference.

The metadata tells the application which features are expected and what
classes the model knows.

## 27. Why Joblib?

Joblib is commonly used to serialize Python/scikit-learn objects.

It allows the application to:

-   Save a trained model
-   Load it later
-   Avoid retraining during normal inference
-   Reuse the trained model in deployment

# 28. Application technology stack

The current repository uses:

``` text
Python
├── Pandas
├── Scikit-learn
├── Joblib
├── Streamlit
├── SHAP
└── Matplotlib
```

GitHub is used to store/manage the project repository.

The current web interface is Streamlit-based; it is not a React
frontend.

# 29. Why Python?

Python is used for:

-   Data handling
-   Input validation
-   Model loading
-   Model inference
-   Probability calculation
-   SHAP analysis
-   Visualization
-   Web UI

# 30. Why Pandas?

Pandas handles tabular data.

In this project it is used for:

-   Reading CSV files
-   Creating DataFrames
-   Preparing model inputs
-   Numeric conversion
-   Input validation
-   Probability tables
-   Explanation tables

A DataFrame is a two-dimensional table of rows and columns.

# 31. Why Streamlit?

Streamlit allows an interactive data/ML web application to be built
directly in Python.

Instead of building:

``` text
React + HTML + CSS + JavaScript + separate backend
```

the current project uses:

``` text
Python
   ↓
Streamlit
   ↓
Web application
```

# 32. Why Matplotlib?

Matplotlib is used for visualizing model output, such as probability
charts.

# 33. Why SHAP?

SHAP makes model predictions more interpretable.

It answers:

> Which features contributed most strongly to this prediction?

# 34. Application startup flow

When the application starts:

``` text
app.py starts
     ↓
Page configuration
     ↓
Find application directory
     ↓
Find model path
     ↓
Check model file
     ↓
Load model with Joblib
     ↓
Extract model metadata
     ↓
Build UI
```

# 35. `Path(__file__).parent`

The application uses the directory containing `app.py` to build the
model path.

Conceptually:

``` python
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "steel_fault_model_bundle.pkl"
```

This avoids hardcoding a personal Windows path.

# 36. Model loading and caching

The application uses `st.cache_resource` around model loading.

Why?

Streamlit can rerun application code during interaction. The model
should not be unnecessarily reloaded every time.

``` text
First run
   ↓
Load model
   ↓
Cache model

Later interactions
   ↓
Reuse model
```

# 37. Error handling

The application checks whether the model file exists and handles
model-loading errors.

This prevents a missing/corrupt model from producing an unexplained
failure.

# 38. Sample input

The repository contains:

``` text
steel_app_example_input.csv
```

It provides an example row using the expected feature schema.

This allows a user to test the application without manually entering all
27 values.

# 39. Manual input

The application provides numeric input controls for the 27 features.

The user enters the measurements and submits the prediction form.

# 40. CSV input

The application allows a CSV upload.

The flow is:

``` text
User uploads CSV
       ↓
Pandas reads CSV
       ↓
Validate required columns
       ↓
Use first row
       ↓
Validate numeric values
       ↓
Create input DataFrame
       ↓
Predict
```

# 41. CSV schema validation

The application checks whether required feature columns are present.

If a required feature is missing:

``` text
CSV
 ↓
Missing required column
 ↓
Show error
 ↓
Do not make prediction
```

This is called **schema validation**.

# 42. Numeric validation

The application uses numeric conversion with invalid values coerced for
detection.

Conceptually:

``` text
"500"    → 500
"12.5"   → 12.5
"hello"  → invalid/NaN
```

This prevents malformed values from being silently passed to the model.

# 43. Current CSV limitation

The current application uses only the first row of an uploaded CSV.

Therefore:

``` text
100 rows uploaded
       ↓
Current app
       ↓
First row used
       ↓
One prediction
```

A future version could implement true batch inference.

# 44. Creating the model input

The application creates a Pandas DataFrame containing one row and the
expected feature columns.

Conceptually:

``` text
Input shape = 1 × 27
```

Keeping the correct feature schema/order is important.

# 45. Prediction

The key inference operation is:

``` python
prediction = model.predict(input_df)[0]
```

Meaning:

> Give the new 27-feature input to the trained classifier and obtain the
> predicted class.

Example:

``` text
27 new measurements
        ↓
model.predict()
        ↓
"Bumps"
```

# 46. Probability prediction

The application also uses:

``` python
probabilities = model.predict_proba(input_df)[0]
```

This returns estimated probabilities for the known classes.

Example:

``` text
Bumps          82%
Other_Faults   10%
Pastry           3%
Stains           2%
K_Scatch         2%
Dirtiness        1%
Z_Scratch        0%
```

The exact values depend on the input.

# 47. `predict()` vs `predict_proba()`

`predict()`:

``` text
What class?
→ Bumps
```

`predict_proba()`:

``` text
How is probability distributed across classes?
→ Bumps 0.82, ...
```

Probability is an estimate, not a guarantee of correctness.

# 48. SHAP

SHAP means:

> **SHapley Additive exPlanations**

It is used to explain model predictions.

The application uses:

``` python
shap.TreeExplainer(model)
```

and calculates SHAP values for the prediction.

# 49. Why TreeExplainer?

TreeExplainer is designed for tree-based models and is appropriate for
explaining the deployed tree model.

# 50. SHAP flow

``` text
New input
    ↓
Model prediction
    ↓
Identify predicted class
    ↓
TreeExplainer
    ↓
SHAP values
    ↓
Select explanation for predicted class
    ↓
Match values to feature names
    ↓
Rank by absolute magnitude
    ↓
Display top contributors
```

# 51. Positive and negative SHAP values

Conceptually:

-   Positive SHAP contribution: pushes the explained output toward the
    class.
-   Negative SHAP contribution: pushes it away.

The exact interpretation depends on the output being explained.

# 52. Why absolute SHAP values?

Suppose:

``` text
Feature A = +0.80
Feature B = -0.75
Feature C = +0.05
```

Magnitude:

``` text
A = 0.80
B = 0.75
C = 0.05
```

The application can therefore identify which features had the largest
impact magnitude.

# 53. SHAP does not prove causation

If SHAP says:

``` text
Pixels_Areas
```

strongly influenced a prediction, it means the model relied on that
feature for its prediction.

It does not prove:

> Pixels_Areas physically caused the defect.

Correct:

> "This feature was influential in the model's prediction."

Incorrect:

> "This feature caused the defect."

# 54. Process Optimization Suggestions

The application contains process-monitoring suggestions based on model
feature importance and observed confusion patterns.

It highlights measurements such as:

-   Length_of_Conveyer
-   LogOfAreas
-   Pixels_Areas
-   Outside_X_Index
-   Sum_of_Luminosity
-   Orientation_Index
-   Minimum_of_Luminosity
-   Log_X_Index

The idea is to help quality teams monitor influential measurements.

These are decision-support suggestions, not causal manufacturing
conclusions.

# 55. Confidence and human inspection

ML predictions should not always be treated as unquestionable.

For uncertain predictions:

``` text
Low confidence
     ↓
Manual inspection
```

This creates a human-in-the-loop quality-control workflow.

# 56. Complete runtime flow

``` text
                 STEEL PLATE
                     │
                     ▼
              Defect detected
                     │
                     ▼
             27 measurements
                     │
                     ▼
              Streamlit UI
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      Manual input          CSV upload
          │                     │
          └──────────┬──────────┘
                     ▼
              Input validation
                     │
                     ▼
              Pandas DataFrame
                     │
                     ▼
                ML model
                     │
             ┌───────┴────────┐
             ▼                ▼
         predict()       predict_proba()
             │                │
             ▼                ▼
       Fault category    Probabilities
             │
             ▼
            SHAP
             │
             ▼
      Feature contributions
             │
             ▼
       Process suggestions
             │
             ▼
           User result
```

# 57. Complete training-to-deployment flow

``` text
UCI Steel Plates Faults Dataset
            │
            ▼
       Data preparation
            │
            ▼
      Model training
            │
            ▼
     Trained classifier
            │
            ▼
       Joblib bundle
            │
            ▼
     steel_fault_model_bundle.pkl
            │
            ▼
        Streamlit app
            │
            ▼
       New user input
            │
            ▼
        Validation
            │
            ▼
        Prediction
            │
       ┌────┴────┐
       ▼         ▼
 Probability   SHAP
       │         │
       └────┬────┘
            ▼
      Decision support
```

# 58. What the project does NOT do

Do not claim that the current application:

-   Directly processes steel-plate photographs
-   Automatically controls factory machinery
-   Uses physical sensors itself
-   Guarantees prediction correctness
-   Automatically retrains itself
-   Predicts simply "good" vs "bad"
-   Processes every row of an uploaded CSV
-   Proves the physical cause of a defect

The current system takes numerical defect measurements and classifies
them.

# 59. Current project limitations

1.  CSV interface currently uses only the first row.
2.  The model knows only its trained fault categories.
3.  The application does not directly process images.
4.  No automatic factory-machine control is implemented.
5.  No automatic retraining pipeline is implemented.
6.  No complete production model-monitoring system is implemented.
7.  Model probabilities are estimates, not guarantees.
8.  The repository does not contain the original training
    notebook/experiment logs.

# 60. Production improvements

A production version could add:

### Batch inference

Process every row of a CSV.

### API

Use FastAPI to expose model predictions through a REST API.

### Containerization

Use Docker.

### Model registry

Use MLflow or another model registry.

### Monitoring

Track:

-   Prediction distribution
-   Model performance
-   Input quality
-   Data drift
-   Confidence
-   Error rates

### Retraining

``` text
New labeled data
       ↓
Validation
       ↓
Training
       ↓
Evaluation
       ↓
Compare with current model
       ↓
Deploy if better
```

### Out-of-distribution detection

Detect inputs that are very different from training data.

### Human-in-the-loop

Send low-confidence or unknown cases to an inspector.

# 61. Data Drift and Concept Drift

## Data drift

The input distribution changes.

Example:

``` text
Old production data
      ↓
New machine/process
      ↓
Different feature distribution
```

## Concept drift

The relationship between input features and the target changes.

Both can cause model performance to degrade.

A production ML system should monitor for these problems.

# 62. New Fault Category

Suppose a completely new defect type appears.

The current seven-class model does not automatically learn it.

A better workflow is:

``` text
Unknown/uncertain case
       ↓
Human inspection
       ↓
Label new fault
       ↓
Collect examples
       ↓
Retrain/update model
       ↓
Evaluate
       ↓
Deploy
```

# 63. Project repository

The current repository contains:

``` text
steel-plate-quality-prediction/
│
├── app.py
├── requirements.txt
├── steel_app_example_input.csv
└── steel_fault_model_bundle.pkl
```

## `app.py`

Main Streamlit application.

Handles:

-   UI
-   Model loading
-   Manual input
-   CSV input
-   Validation
-   Prediction
-   Probability output
-   SHAP explanation
-   Process suggestions

## `requirements.txt`

Python dependencies.

## `steel_app_example_input.csv`

Example input.

## `steel_fault_model_bundle.pkl`

Serialized trained model bundle.

# 64. Strong 30-second interview explanation

> "I developed an AI-based steel plate fault classification application.
> The system uses the Steel Plates Faults dataset containing 27 input
> features and seven fault categories. I trained a tree-based
> classification model and serialized it using Joblib. I then built a
> Streamlit application where users can enter the 27 measurements
> manually or upload a CSV. The application validates the input, creates
> a Pandas DataFrame, passes it to the trained model, and displays the
> predicted fault category and class probabilities. I also integrated
> SHAP's TreeExplainer to explain which features contributed most to
> individual predictions. The goal is to provide a machine-learning
> decision-support tool for manufacturing quality control."

# 65. Strong 2-minute interview explanation

> "My project is an AI-based steel plate quality-control application
> designed to classify surface defects.
>
> The problem is that steel plates can develop different types of
> surface faults during manufacturing. The current application uses
> numerical measurements describing a detected defect rather than
> directly processing a photograph.
>
> I used the Steel Plates Faults dataset, which contains 1,941 records,
> 27 input features and seven fault categories: Pastry, Z_Scratch,
> K_Scatch, Stains, Dirtiness, Bumps and Other_Faults.
>
> The 27 features describe characteristics such as position, area,
> perimeter, luminosity, shape, edge characteristics, steel type and
> plate thickness. Therefore, this is a multiclass classification
> problem.
>
> I trained a tree-based classification model and serialized the trained
> model into a Joblib model bundle. The bundle also stores the expected
> feature columns and target classes so the deployed application knows
> the expected input schema.
>
> I built a Streamlit application for inference. Users can enter the 27
> measurements manually, load a sample input, or upload a CSV. The
> application validates the input schema and numeric values, creates a
> Pandas DataFrame with the expected feature order, and passes it to the
> trained model.
>
> The model's `predict()` method returns the fault category. I also use
> `predict_proba()` to show the estimated probability distribution
> across the seven classes.
>
> For explainability, I integrated SHAP's TreeExplainer. This helps show
> which features contributed most strongly to an individual prediction.
> I use this as model interpretation, not as proof of causation.
>
> The application also provides process-monitoring suggestions based on
> model feature importance and observed confusion patterns. The system
> is intended as decision support rather than a replacement for physical
> inspection.
>
> One current limitation is that the CSV interface uses only the first
> row. For production, I would add batch prediction, model monitoring,
> drift detection, model versioning, an API layer, and an automated
> retraining pipeline."

# 66. High-priority interview questions

Be able to answer these without notes:

1.  Explain your project.
2.  What problem does it solve?
3.  Is it hardware or software?
4.  What dataset did you use?
5.  How many records?
6.  How many features?
7.  What are the 27 features?
8.  What are the seven classes?
9.  What is the input?
10. What is the output?
11. Why is it classification?
12. Why multiclass?
13. What algorithm did you use?
14. Why Random Forest?
15. What is a decision tree?
16. What is Random Forest?
17. What is bagging?
18. What is Gini impurity?
19. What is overfitting?
20. What is generalization?
21. What is accuracy?
22. Precision vs recall?
23. What is F1?
24. What is a confusion matrix?
25. Why Python?
26. Why Pandas?
27. Why Streamlit?
28. Why Joblib?
29. What is a `.pkl` file?
30. What is serialization?
31. What does `predict()` do?
32. What does `predict_proba()` do?
33. What happens when CSV is uploaded?
34. How do you validate input?
35. What happens if a column is missing?
36. What happens if a value is non-numeric?
37. Why cache the model?
38. What is SHAP?
39. Why SHAP?
40. What is TreeExplainer?
41. What does a positive SHAP value mean?
42. Does SHAP prove causation?
43. What are the project limitations?
44. How would you support batch prediction?
45. How would you monitor the model?
46. What is data drift?
47. What is concept drift?
48. How would you retrain the model?
49. How would you deploy it at factory scale?
50. What would you improve?

# 67. Final mental model

Memorize this:

``` text
HISTORICAL STEEL DATA
        ↓
27 measurements + labels
        ↓
MODEL TRAINING
        ↓
TRAINED MODEL
        ↓
JOBLIB MODEL BUNDLE
        ↓
STREAMLIT APPLICATION
        ↓
NEW 27 INPUTS
        ↓
VALIDATION
        ↓
1 × 27 DATAFRAME
        ↓
TRAINED MODEL
        ↓
 ┌───────────────┬───────────────┐
 ↓               ↓               ↓
predict()    predict_proba()    SHAP
 ↓               ↓               ↓
Fault         Probabilities    Explanation
 └───────────────┴───────────────┘
                 ↓
        QUALITY-CONTROL
        DECISION SUPPORT
```

# 68. One-sentence final answer

> **I built a Python and Streamlit-based explainable machine-learning
> application that uses 27 numerical characteristics of steel-plate
> defects to classify them into seven fault categories, provides
> probability estimates, and uses SHAP to explain the features
> influencing each prediction.**

------------------------------------------------------------------------

## Critical interview rule

Be confident about features that are actually implemented:

-   Python
-   Pandas
-   Scikit-learn/tree-based classification
-   Joblib
-   Streamlit
-   SHAP
-   CSV validation
-   Probability output
-   Model explanations

Do not invent:

-   Exact accuracy
-   Exact train/test split
-   Exact hyperparameters
-   Exact preprocessing steps
-   Exact confusion matrix
-   Technologies you did not use

If an interviewer asks for a detail that is not available in the current
deployment repository, say:

> "That detail belongs to the original training experiment; the current
> repository contains the serialized model and deployment code rather
> than the original training notebook and experiment logs. I would refer
> to the original training experiment for the exact value."

This is much better than guessing.
