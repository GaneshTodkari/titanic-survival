🚢 Titanic - Machine Learning from Disaster
Predict survival on the Titanic and get familiar with ML basics
📌 Problem Statement
The sinking of the Titanic was one of the deadliest shipwrecks in history. The goal of this machine learning project is to predict which passengers survived the tragedy, based on information such as age, gender, class, and fare paid.

🗂 Dataset Description
The dataset is provided by Kaggle's Titanic Competition. It includes:

train.csv – labeled data to train the model (891 rows)

test.csv – unlabeled data for predictions (418 rows)

gender_submission.csv – sample submission format

Key Columns:
Column	Description
PassengerId	Unique ID
Survived	0 = No, 1 = Yes (Target variable)
Pclass	Ticket class (1 = Upper, 3 = Lower)
Name	Full name
Sex	Gender
Age	Age in years
SibSp	Siblings/spouses aboard
Parch	Parents/children aboard
Ticket	Ticket number
Fare	Passenger fare
Cabin	Cabin number
Embarked	Port of Embarkation (C, Q, S)

🔍 Data Preprocessing & Feature Engineering
Steps performed:

Dropped irrelevant features: Cabin, Ticket, Name

Imputed missing values in Age, Embarked, Fare

Encoded categorical variables: Sex, Embarked

Created new features:

FamilySize = SibSp + Parch + 1

IsAlone = 1 if FamilySize == 1

Title extracted from Name

AgeBin, FareBin using quantile binning

🤖 Models Used
Logistic Regression with GridSearchCV

Random Forest Classifier with hyperparameter tuning

K-Nearest Neighbors

Naive Bayes


🧪 Model Evaluation
Models were evaluated using:

Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

Cross-validation (CV = 5 folds)

📤 Submission
Predictions were made on test.csv, and results were saved in Result.csv with the format:

PassengerId,Survived
892,0
893,1
...
1309,0
Submitted to Kaggle Titanic competition.

✅ Best Score
Kaggle Public Leaderboard Score: 0.77033

Achieved using Random Forest + feature engineering

📂 Folder Structure
kotlin
Copy
Edit
📁 Titanic-ML
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── notebooks/
│   └── titanic_model.ipynb
├── results/
│   └── Result.csv
├── README.md
└── requirements.txt
🚀 How to Run
Clone the repo:

git clone https://github.com/GaneshTodkari/titanic-survival.git
cd titanic-ml
Install dependencies:

pip install -r requirements.txt
Run the notebook:

jupyter notebook notebooks/titanic_model.ipynb
📘 Learnings & Takeaways
Feature engineering can significantly boost model performance.

Ensemble models like RandomForest and VotingClassifier are powerful for classification.

Proper preprocessing (handling nulls, encoding, scaling) is crucial.

🙌 Acknowledgements
Kaggle Titanic dataset

scikit-learn, pandas, seaborn

Community notebooks and discussions