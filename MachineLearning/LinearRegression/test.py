import tensorflow
import keras
import sklearn
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
data = pd.read_csv("student-mat.csv", sep = ';')
data = data[["G1", "G2", "G3", "goout", "studytime", "Dalc", "health"]]
#traveltime : 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour

predict = "G3"
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
# best = 0
# for i in range(130):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.3)
#
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
#     if acc > best:
#         best = acc
#         with open("studentmodel.pickle", "wb") as f:
#             pickle.dump(linear, f)
#         print(acc)


pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Co: " ,  linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)
acc = linear.score(x_test, y_test)
for x in range(len(predictions)):
    print(predictions[x],  y_test[x])
print(acc)
p = "goout"
# style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
# print("acc: ", acc)
