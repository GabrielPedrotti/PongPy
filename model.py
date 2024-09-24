import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

df = pd.read_csv('game_data.csv')

X = df[['ball_x', 'ball_y', 'left_pad_y', 
         'ball_dx', 'ball_dy', 'left_pad_distance',
         'ball_distance_to_left_pad', 'ball_direction']]
y = df['player_action']

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = xgb.XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy * 100:.2f}%")
joblib.dump(model, 'pong_model.pkl')
joblib.dump(encoder, 'encoder.pkl')

print("Modelo salvo com sucesso!")
