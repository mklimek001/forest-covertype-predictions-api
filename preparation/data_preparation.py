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

X_valid = df_raw.iloc[11340: 11340 + 3780].copy()
y_valid = X_valid.pop('Cover_Type')

# data normalization
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_normalized = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_valid_normalized = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)

joblib.dump(scaler, '../models/scaler.save')

train = X_train_normalized.copy()
valid = X_valid_normalized.copy()

train['Cover_Type'] = y_train.values
valid['Cover_Type'] = y_valid.values

train.to_csv('../data/train.csv')
valid.to_csv('../data/valid.csv')


# preparing dataset for tests (same number of samples from every class)
potentially_to_test = df_raw.iloc[11340 + 3780:]
minimal = len(df_raw)

for i in range(1, 8):
    minimal = min(len(potentially_to_test[potentially_to_test['Cover_Type'] == i]), minimal)

test_selection = []

for i in range(1, 8):
    test_selection.append(potentially_to_test[potentially_to_test['Cover_Type'] == i][:minimal].copy())

test = pd.concat(test_selection)
test.to_csv('../data/test.csv')
