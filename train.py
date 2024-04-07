import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)  # Score - 0.428

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
