#Import Data Manipulation Libraries
import pandas as pd
import numpy as np

# Import Data Visualisation Libraries
import matplotlib.pyplot as plt
import seaborn as sns


# Import machine learning libraries
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 

# Import Warningg Libraries
import warnings
warnings.filterwarnings('ignore')

# Import Data Logging Libraries

import logging
logging.basicConfig( level=logging.INFO,
                    filename='log.txt',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s ',force = True)

url = 'https://raw.githubusercontent.com/Arbajshaikh7030/Cropp_Recommendation_model/refs/heads/main/Crop_Recommendation.csv'

df = pd.read_csv(url)

# Split the dataset into X and y 

X = df.drop(['Crop','Rainfall'], axis = 1)
y = df['Crop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))