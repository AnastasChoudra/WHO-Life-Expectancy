# load libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# load dataset
dataset = pd.read_csv('life_expectancy.csv')

# Data loading and observing
print(dataset.head())
#print(dataset.describe())
dataset = dataset.drop(columns='Country', axis=1)

labels = dataset.iloc[:,-1] 
features = dataset.drop(columns='Life expectancy', axis=1)

# Data Preprocessing
# Transform features into categorical with dummy coding (Hot Encoding)
features = pd.get_dummies(features)
# create training and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=1)
# standardize features
# make features numerical
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')
# fit training data
features_train_scaled = ct.fit_transform(features_train)
# transform test data
features_test_scaled = ct.transform(features_test)

# create neural model
my_model = Sequential()
# creaste input layer
input = InputLayer(input_shape=(features.shape[1],))
# add layer to model
my_model.add(input)
# add hidden layer
my_model.add(Dense(128, activation='relu'))
my_model.add(Dense(1)) # just one output for regression
# print results
print(my_model.summary())

# optimize model
opt = Adam(learning_rate=0.01)
# compile
my_model.compile(loss='mse', metrics=['mae'], optimizer=opt)
# train model
my_model.fit(features_train_scaled, labels_train, epochs=40, batch_size=1, verbose=1)
# evaluate model
res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose=0)

# print results
print("RMSE: ", res_mse)
print("MAE: ", res_mae)