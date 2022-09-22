import os
from dotenv import load_dotenv
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import pickle
from pathlib import Path

load_dotenv()

RANDOM_STATE = int(os.getenv("RANDOM_STATE"))
RESAMPLING = int(os.getenv("RESAMPLING"))
CACHE = os.getenv("CACHE")

def load():
    return joblib.load(Path(CACHE,'pipe.pkl'))

def save(pipe):

    if not Path(CACHE).exists():
        Path(CACHE).mkdir()

    joblib.dump(pipe, Path(CACHE,'pipe.pkl'), compress = 1)

def build(df, features=[], target="", test_size=0.3, autosave=True):
    

    ml_model = SVC()

    X = np.concatenate(df.beat.values).reshape(-1, RESAMPLING)
    y = df[target]
  
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        random_state=RANDOM_STATE,
        test_size=test_size)

    pipe = Pipeline(steps=[
            ('model', ml_model),
        ]
    )

    pipe.fit(X_train, y_train)

    score = pipe.score(X_test, y_test)

    if autosave:
        save(pipe)

    return pipe