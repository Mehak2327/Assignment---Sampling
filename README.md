# Sampling Assignment – Handling Imbalanced Credit Card Dataset

Student Name: Mehak  
Roll Number: 102303792  

---

## 📌 Objective

The objective of this assignment is to study the importance of sampling techniques in handling highly imbalanced datasets and to analyze how different sampling strategies affect the performance of various machine learning models.

---

## 📂 Dataset

The dataset used in this project is the Credit Card Fraud Detection dataset provided by the instructor.

Dataset Link:  
https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv  

The dataset contains transaction records with a binary class label:  
- 0 → Normal Transaction  
- 1 → Fraudulent Transaction  

Since fraudulent cases are very few, the dataset is highly imbalanced.

---

## ⚙️ Methodology

The following steps were followed:

1. Load the dataset and separate features (X) and target variable (y).
2. Apply five different sampling techniques to balance the dataset:
   - Sampling1: Random Over Sampling
   - Sampling2: Random Under Sampling
   - Sampling3: SMOTE (Synthetic Minority Over-sampling Technique)
   - Sampling4: SMOTEENN
   - Sampling5: ADASYN
3. Split each sampled dataset into training (70%) and testing (30%).
4. Train five machine learning models on each sampled dataset:
   - M1: Logistic Regression  
   - M2: Decision Tree  
   - M3: Random Forest  
   - M4: K-Nearest Neighbors (KNN)  
   - M5: Support Vector Machine (SVM)
5. Evaluate each model using Accuracy Score.
6. Store and compare the results in tabular and graphical form.

---

## 🧠 Machine Learning Models Used

- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- KNN Classifier  
- Support Vector Machine (SVM)

---

## 📊 Result Table

| Model | Sampling1 (ROS) | Sampling2 (RUS) | Sampling3 (SMOTE) | Sampling4 (SMOTEENN) | Sampling5 (ADASYN) |
|-----|-----|-----|-----|-----|-----|
| M1 Logistic Regression | 91.92 | 83.33 | 92.14 | 94.19 | 91.94 |
| M2 Decision Tree | 98.47 | 83.33 | 98.03 | 98.84 | 97.17 |
| M3 Random Forest | 99.78 | 16.67 | 99.34 | 99.71 | 98.91 |
| M4 KNN | 98.47 | 33.33 | 85.59 | 96.51 | 83.44 |
| M5 SVM | 70.31 | 16.67 | 70.52 | 73.84 | 67.97 |

---

## 🏆 Best Sampling Technique for Each Model

| Model | Best Sampling Technique | Accuracy |
|-----|------------------------|----------|
| Logistic Regression | SMOTEENN | 94.19 |
| Decision Tree | SMOTEENN | 98.84 |
| Random Forest | SMOTEENN | 99.71 |
| KNN | Random Over Sampling | 98.47 |
| SVM | SMOTEENN | 73.84 |

---

## 📈 Result Graph

A bar graph is plotted to visualize the accuracy comparison between different sampling techniques and models.
See **result_graph.png** in the repository for graphical comparison of results.

---

## 📌 Conclusion

- Sampling techniques significantly impact model performance.
- SMOTEENN provided the best performance for most models.
- Random Over Sampling worked well for KNN.
- Random Forest achieved the highest overall accuracy.

---

## 🛠 Tools & Technologies

- Python  
- Google Colab  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn  
- Matplotlib  

---

## 📁 Repository Structure

Sampling_Assignment  
│── Sampling_Assignment.py  
│── README.md  
│── result_graph.png  

---

## ▶ How to Run

1. Open the notebook in Google Colab.  
2. Run all cells sequentially.  
3. Results and graphs will be generated automatically.

---

