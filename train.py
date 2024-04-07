import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

mod = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=25)
model = mod.fit(X,y)

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
