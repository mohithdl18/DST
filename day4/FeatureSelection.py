#types of feature selection
#Filter selection - correlation matrix
#Wrapper selection - 
#Embedding selection
#Dimensionality reduction

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import kagglehub

# Download latest version
path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")

print("Path to dataset files:", path)

# Find the CSV file within the downloaded directory
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        csv_file_path = os.path.join(path, filename)
        break  # Stop after finding the first CSV file

# Read the CSV file using the corrected file path
df = pd.read_csv(csv_file_path) 

# Example prepocessing (scaling)
# Separate features (X) and target (y)
X = df.drop('age', axis=1)  # Use capital X for features
y = df['age']

# Create a StandardScaler instance
scaler = StandardScaler()  # Assign scaler to a different variable

# Fit and transform the features
X_scaled = scaler.fit_transform(X)  # Apply scaling to the features data

# Initialize logistic regression with max_iter increased
model = LogisticRegression(max_iter=1000)
# Apply RFE
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(X_scaled, y)  # Use scaled features for RFE

print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)



# forward and bacjward feature selection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
sfs = SFS(model, k_features=5, forward=True, verbose=2, scoring='accuracy', cv=5)
sfs = sfs.fit(df.drop('target', axis=1), df['target'])
print('Best accuracy score: %.2f' % sfs.k_score_)
print('Best subset (indices):', sfs.k_feature_idx_)
print('Best subset (corresponding names):', sfs.k_feature_names_)

