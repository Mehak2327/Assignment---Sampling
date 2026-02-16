import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# Load dataset
url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
data = pd.read_csv(url)

X = data.drop("Class", axis=1)
y = data["Class"]

# Sampling techniques
sampling_methods = {
    "Sampling1_RandomOver": RandomOverSampler(),
    "Sampling2_RandomUnder": RandomUnderSampler(),
    "Sampling3_SMOTE": SMOTE(),
    "Sampling4_SMOTEENN": SMOTEENN(),
    "Sampling5_ADASYN": ADASYN()
}

# Models
models = {
    "M1_Logistic": LogisticRegression(max_iter=10000),
    "M2_DecisionTree": DecisionTreeClassifier(),
    "M3_RandomForest": RandomForestClassifier(),
    "M4_KNN": KNeighborsClassifier(),
    "M5_SVM": SVC()
}

results = {}

for s_name, sampler in sampling_methods.items():
    X_res, y_res = sampler.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, random_state=42
    )
    
    results[s_name] = {}
    for m_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[s_name][m_name] = round(acc*100,2)

result_df = pd.DataFrame(results)
print(result_df)

# GRAPH PLOTTING 

result_df = result_df.T   

plt.figure()
result_df.plot(kind="bar")
plt.xlabel("Sampling Techniques")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison of Sampling Techniques vs ML Models")
plt.legend(title="Models", bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.show()
