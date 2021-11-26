from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, f1_score, precision_score

#X and y should be defined from the data.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50), alpha=0.0001, solver='adam', random_state=3)
mlp.fit(X_train, y_train)
y_res = mlp.predict(X_test)
print("accuracy:", mlp.score(X_test, y_test))
print("recall: ", recall_score(y_res, y_test))
print("f1: ", f1_score(y_res, y_test))
print("precision: ", precision_score(y_res, y_test))

# This part of fitting and predicting will be used later with our database to make predictions for the optimization model.
