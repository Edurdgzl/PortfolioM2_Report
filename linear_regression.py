import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('Fish.csv')

print("\n\n\nTESTS\n\n\n")
print("TEST 1")
print("-------------------------------------------------------------------------------------------------------------------")
y = df['Weight']
x = df[['Length1', 'Height','Width']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)
regressor = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, alpha=0.0001))
regressor.fit(X_train, y_train)
print("\nValues of m: {} ".format(regressor.steps[1][1].coef_))
print("\nValue of b: {} ".format(regressor.steps[1][1].intercept_))



print("\nCoefficient of determination of the prediction. (Score): {}".format(regressor.score(X_test, y_test)))

print("\n\n\nPREDICTION 1")
print("-------------------------------------------------------------------------------------------------------------------")
y_pred = regressor.predict(X_test)

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\n")
print(results)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("\nMean absolute error: {:.2f}".format(mae))
print("\nMean squared error: {:.2f}".format(mse))
print("\nRoot mean squared error: {:.2f}".format(rmse))
""" plt.scatter(X_test.iloc[:, 1], y_test,color='g')
plt.scatter(X_test.iloc[:, 1], y_pred,color='b',s=9)
plt.legend(["Valor actual de y" , "Predicción"])
plt.title("Valor actual de y contra predicción")
plt.show() """
print("-------------------------------------------------------------------------------------------------------------------")

print("\n\n\nTEST 2")
print("-------------------------------------------------------------------------------------------------------------------")
y = df['Weight']
x = df[['Length1', 'Height','Width']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

regressor = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, alpha=0.0001))
regressor.fit(X_train, y_train)
print("\nValues of m: {} ".format(regressor.steps[1][1].coef_))
print("\nValue of b: {} ".format(regressor.steps[1][1].intercept_))



print("\nCoefficient of determination of the prediction. (Score): {}".format(regressor.score(X_test, y_test)))

print("\n\n\nPREDICTION 2")
print("-------------------------------------------------------------------------------------------------------------------")
y_pred = regressor.predict(X_test)

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\n")
print(results)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("\nMean absolute error: {:.2f}".format(mae))
print("\nMean squared error: {:.2f}".format(mse))
print("\nRoot mean squared error: {:.2f}".format(rmse))
""" plt.scatter(X_test.iloc[:, 1], y_test,color='g')
plt.scatter(X_test.iloc[:, 1], y_pred,color='b',s=9)
plt.legend(["Valor actual de y" , "Predicción"])
plt.title("Valor actual de y contra predicción")
plt.show() """
print("-------------------------------------------------------------------------------------------------------------------")

print("\n\n\nTEST 3")
print("-------------------------------------------------------------------------------------------------------------------")
y = df['Weight']
x = df[['Length1', 'Height','Width']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=60)

regressor = make_pipeline(StandardScaler(), SGDRegressor(max_iter=100, alpha=0.001))
regressor.fit(X_train, y_train)
print("\nValues of m: {} ".format(regressor.steps[1][1].coef_))
print("\nValue of b: {} ".format(regressor.steps[1][1].intercept_))



print("\nCoefficient of determination of the prediction. (Score): {}".format(regressor.score(X_test, y_test)))

print("\n\n\nPREDICTION 3")
print("-------------------------------------------------------------------------------------------------------------------")
y_pred = regressor.predict(X_test)

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\n")
print(results)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("\nMean absolute error: {:.2f}".format(mae))
print("\nMean squared error: {:.2f}".format(mse))
print("\nRoot mean squared error: {:.2f}".format(rmse))
""" plt.scatter(X_test.iloc[:, 1], y_test,color='g')
plt.scatter(X_test.iloc[:, 1], y_pred,color='b',s=9)
plt.legend(["Valor actual de y" , "Predicción"])
plt.title("Valor actual de y contra predicción")
plt.show() """
print("-------------------------------------------------------------------------------------------------------------------")
