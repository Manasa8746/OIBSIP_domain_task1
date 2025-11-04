# OIBSIP_domain_task1
“Task 1 project for Oasis Infobyte Internship”
🌸 Project Title: Iris Flower Classification using Machine Learning.

🎯 Objective:
   To develop a machine learning model that classifies iris flowers into three species — Setosa, Versicolor, and Virginica — based on their sepal length, sepal width, petal   length, and petal width.
   This project helps understand how supervised learning algorithms learn patterns from data to make accurate predictions.
   
⚙️ Steps Performed:
Imported necessary Python libraries (Pandas, NumPy, Scikit-learn).
Loaded the Iris dataset from a CSV file using Pandas.
Mapped target labels to corresponding species names.
Split data into training (80%) and testing (20%) sets.
Normalized data using StandardScaler() to ensure uniform feature scaling.
Trained a Logistic Regression model on the training dataset.
Predicted flower species on the test dataset.
Evaluated performance using accuracy score, confusion matrix, and classification report.
Tested the model with a sample input for new prediction.

🧰 Tools Used:
Python
Pandas → For data handling and preprocessing
NumPy → For numerical operations
Scikit-learn (sklearn) → For model training and evaluation
Logistic Regression → For classification
Jupyter Notebook / VS Code → For code execution and testing.

📊 Output / Results:
The model achieved 100% accuracy on the test dataset.
The confusion matrix and classification report confirmed correct predictions for all three species.
A sample flower measurement was tested successfully, and the model predicted the correct species — Setosa.

Example Output:
🔹 Model Accuracy: 1.0
🔹 Confusion Matrix:
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
🔹 Predicted species for sample input: setosa

