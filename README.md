# Wine_Quality_Prediction
This project aims to develop an analytical classification model to predict the quality of the red variant of the Portuguese “Vinho Verde” style of wine. The data comes from the UCI Machine Learning Repository. The classification model will provide insights into the quality of the wine by predicting whether it is good (1) or bad (0).

# Wine Quality Data:
Data Set Characteristics: Multivariate
Number of Attributes: 11 + output attribute
Number of Records: 1599
Number of Missing Values: 0
Data donated: 2009-10-07

# Data Attributes:
Fixed Acidity
Volatile Acidity
Citric Acid
Residual Sugar
Chlorides
Free sulfur dioxide
Total Sulfur Dioxide
Density
pH
Sulphates
Alcohol
Quality (output variable)

# Data Quality Concerns:
● The different classes of wine are not balanced and are unevenly represented in the data; average-quality wines outweigh both excellent and poor wines. This imbalance will be addressed in the final model using the SMOTE sampling technique.

● None of the features are known to hold more importance than others. Feature selection will need to take place to determine which variables are most important to the model.

# Data Preparation:
Converting Target Variable to Binary

1. "Quality" is added as a new column to the data frame "data" with a value of 1 if the value of the original "quality" column is higher than or equal to 7, and 0 otherwise. Here, the "quality" value is transformed into a binary variable that denotes whether the wine is good (quality >=7) or not (quality < 7).

2. Binned quality rankings to either "good" (7, 8) as 1 or "bad" (3, 4, 5, 6) as 0.

3. This new binning is what the models will predict, either 1 or 0.

# Checking for Class Imbalance

● We examined the training data for evidence of class imbalance. Class imbalance is when there are not an equal number of samples in each class, which might result in biased model performance.

● The output indicates that the training data is skewed toward the negative class (quality = 0), with 1382 samples having a value of 0 and 217 samples having a value of 1.

● The output verifies that the training data is unbalanced since 86.4% of it belongs to the negative class (quality=0) and only 13.5% to the positive class (quality=1).

● To avoid the model being biased toward the dominant class, it is crucial to resolve the class imbalance in the training data. Class imbalance can be addressed using a variety of methods, including undersampling the dominant class, oversampling the minority class, or combining both.

# Data splitting

● Made a new data frame called "x" that has every column from the original data frame called "data," with the exception of the "quality" column.

● Produced a new series "y" that only includes the "quality" column from the original dataframe "data."

● Utilized the scikit-learn train_test_split tool to divide the data into training and testing sets. This is done to provide two distinct datasets that may be utilized for our machine learning model's training and evaluation. The data is divided so that 80% is used for training and 20% is utilized for testing. To ensure that the split can be replicated, a random state of 42 is chosen.

# Handling Class Imbalance using SMOTE

● SMOTE (Synthetic Minority Over-sampling Technique) is an algorithm that analyzes training data. SMOTE is a popular method for creating synthetic samples that are used to
oversample the minority class in unbalanced datasets.

● The results demonstrate that there are now 2218 samples total, with 1109 samples for the negative (quality = 0) and 1109 samples for the positive (quality = 1).

● By giving the model more samples from the minority class to learn from, handling the class imbalance in the training data using SMOTE can enhance the model's performance.

# Data Scaling

● Utilized Scikit-Learn's StandardScaler to scale the data. By doing this, the feature variables are normalized to have a mean of 0 and a standard deviation of 1. This ensures that each feature is of comparable size and prevents any one feature from having an outsized impact on the performance of the model.

# Principal Component Analysis (PCA)

● PCA can assist in reducing the computational complexity of the model and avoiding overfitting by reducing the dimensionality of the data. To prevent losing too much data or adding too much noise to the data, it's crucial to choose the number of principal components carefully.

● n_components = 0.90, which indicates that we want to keep enough principal components to account for at least 90% of the variance in the data, creating an instance of the PCA class.

● Explained Variance ratio determines the overall variance explained by the principal components that were kept after the data were subjected to PCA. The output value of 0.9081 shows that roughly 90% of the variance in the original data is explained by the principal components that were kept.

● Each principal component's explained variance ratio displays the percentage of the overall variance that each component contributes to explaining. The result is an array of seven values, each of which represents a principal component that was kept.

Algorithms Applied:
1. K-Nearest Neighbors (KNN) Classifier
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM) Classifier

# Hyperparameter Tuning:

● To discover the ideal collection of hyperparameters for the above obtained best model, the GridSearchCV function is used for hyperparameter tweaking.

# Best Model:

● A random forest classification model employing the best hyperparameters was found by a grid search on the training data. The model is then applied to forecast the test data, and its performance is assessed using accuracy, precision, recall, and F1 scores.

# Prediction of Quality on a New Test Dataset:

● The most effective random forest classifier model that was trained on a dataset to predict the quality of red wine is used here. On overall, the classifier model's provided an excellent precision and recall scores, in predicting wine quality.

# Bibliography ~

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties. In Decision
Support Systems, Elsevier, 47(4):547-553, 2009.
