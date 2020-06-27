from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

in_data = pd.read_csv('../data/Challenge_9_newFeatures_completed_medium.csv',
                      sep=',',
                      parse_dates=['Baseline Start Date',
                                   'Baseline Finish Date',
                                   'Forecast Start Date',
                                   'Forecast Finish Date'],
                      )
print(in_data.head())

X = in_data.drop('Forecast Duration')
y = np.array(in_data['Forecast Duration'])

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_dict = pd.DataFrame(X_train[:, 1:]).T.to_dict().values()
X_test_dict = pd.DataFrame(X_test[:, :]).T.to_dict().values()
pipe = make_pipeline(DictVectorizer(sparse=False), GradientBoostingRegressor())

pipe.fit(X_train_dict, y_train)
pred = pipe.predict(X_test_dict)
print(r2_score(y_test, pred))