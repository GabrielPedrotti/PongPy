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

# def add_noise(X, scale=2):
#     # Adicionar ruído gaussiano controlado às posições da bola e das raquetes
#     return X + np.random.normal(loc=0, scale=scale, size=X.shape)

# X_noisy = add_noise(X)

 # Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Treinar o modelo usando XGBoost
model = xgb.XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)
# Avaliar o modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy * 100:.2f}%")
# Salvar o modelo treinado
joblib.dump(model, 'pong_model.pkl')
joblib.dump(encoder, 'encoder.pkl')

print("Modelo salvo com sucesso!")
