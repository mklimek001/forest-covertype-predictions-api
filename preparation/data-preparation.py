import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

names = ['Elevation', 'Aspect', 'Slope',
         'Horizontal_Distance_To_Hydrology',
         'Vertical_Distance_To_Hydrology',
         'Horizontal_Distance_To_Roadways',
         'Hillshade_9am',
         'Hillshade_Noon',
         'Hillshade_3pm',
         'Horizontal_Distance_To_Fire_Points']

for i in range(4):
    names.append('Wilderness_Area_' + str(i))

for i in range(40):
    names.append('Soil_Type_' + str(i))

names.append('Cover_Type')

df_raw = pd.read_csv('../data/covtype.data', sep=',', header=None, names=names)

# According to information from covtype.info file, 11340 first values are choosen for test set
# and next 3780 rows are prepared for validation.
# In train and validation set every class occurs in equal numbers.

X_train = df_raw.iloc[: 11340].copy()
y_train = X_train.pop('Cover_Type')

X_valid = df_raw.iloc[11340: 11340+3780].copy()
y_valid = X_valid.pop('Cover_Type')

# data normalization
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_normalized = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test_normalized = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)

joblib.dump(scaler, '../models/scaler.save')

train = X_train.copy()
valid = X_valid.copy()

train['Cover_Type'] = y_train
valid['Cover_Type'] = y_valid

valid.to_csv('../data/valid.csv')
train.to_csv('../data/train.csv')
