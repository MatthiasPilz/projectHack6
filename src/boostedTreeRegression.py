from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

in_data = pd.read_csv('../data/Challenge_9_newFeatures_completed_medium.csv',
                      sep=',',
                      na_values=np.nan,
                      dtype={'Id': np.int32,
                             'ProjectNumber': 'category',
                             'ProductLineNumber': 'category',
                             'ACTIVITY_STATUS': str,
                             'DepartmentNumber': 'category',
                             'ActivityTypeNumber': 'category',
                             'Code': 'category',
                             'ClassNumber': 'category',
                             'Planned Duration': np.int32,
                             'Forecast Duration': np.int32,
                             'Duration Variance': np.int32,
                             'PM_Code_l1_past_mean': np.float64,
                             'PM_Code_l2_past_mean': np.float64,
                             'PM_Code_l3_past_mean': np.float64,
                             'PM_Code_l4_past_mean': np.float64,
                             'Baseline Quarter Start': 'category',
                             'Baseline Start Month': 'category',
                             'Baseline Quarter Finish': 'category',
                             'Baseline Finish Month': 'category',
                             'Forecast Quarter Start': 'category',
                             'Forecast Start Month': 'category',
                             'Forecast Quarter Finish': 'category',
                             'Forecast Finish Month': 'category',
                             'Delayed Start': 'bool',
                             'Delay': 'bool',
                             'Relative Duration Variance': np.float64,
                             },
                      )

print(in_data.loc[:, in_data.isna().any()].head())

print(in_data.dtypes)

print(in_data.head())
print(in_data.columns)
in_data.fillna(0.0, inplace=True)

temp = in_data.drop('Forecast Duration', 1)
temp = temp.drop('ACTIVITY_STATUS', 1)
temp = temp.drop('Id', 1)
temp = temp.drop('Baseline Start Date', 1)
temp = temp.drop('Baseline Finish Date', 1)
temp = temp.drop('Forecast Start Date', 1)
temp = temp.drop('Forecast Finish Date', 1)
temp = pd.get_dummies(temp, prefix='', prefix_sep='')

X = temp.to_numpy()
y = np.array(in_data['Forecast Duration'])

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_dict = pd.DataFrame(X_train[:, 1:]).T.to_dict().values()
X_test_dict = pd.DataFrame(X_test[:, :]).T.to_dict().values()
pipe = make_pipeline(DictVectorizer(sparse=False), GradientBoostingRegressor())

pipe.fit(X_train_dict, y_train)
pred = pipe.predict(X_test_dict)
print(r2_score(y_test, pred))