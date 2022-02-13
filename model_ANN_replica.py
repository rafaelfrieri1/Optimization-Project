import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

dataSet = pd.read_csv('./Database_And_Test_Instances_Feedback_System/3.TrainingTable.csv')

X = dataSet[['Delivery_Norm', 'Shipping_Norm', 'Damage_rate_Norm', 'Population_Norm', 'Employment_Norm', 'Salary_Norm']].values
y = dataSet['Pnum'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 7)

mlp = MLPRegressor(
  hidden_layer_sizes = (16, 16),
  activation = 'relu',
  solver = 'lbfgs',
  learning_rate = 'constant',
  max_iter = 10000,
  random_state = 3,
  n_iter_no_change = 10,
  max_fun = 15000
)

mlp.fit(X_train, y_train)
y_res = mlp.predict(X_test)

print("R2 score:", r2_score(y_test, y_res))
print("Mean squared error:", mean_squared_error(y_test, y_res))