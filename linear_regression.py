import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('Fish.csv')
df1 = df.drop(['Species'], axis=1)
scaler = StandardScaler()
df = scaler.fit_transform(df1)
df = pd.DataFrame(df, columns=df1.columns)
print("TEST")
print("-------------------------------------------------------------------------------------------------------------------")
y = df['Weight']
x = df[['Length1', 'Height','Width']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)
regressor = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, alpha=0.1))
regressor.fit(X_train, y_train)
print("\nValues of m: {} ".format(regressor.steps[1][1].coef_))
print("\nValue of b: {} ".format(regressor.steps[1][1].intercept_))



print("\nCoefficient of determination of the prediction. (Score): {}".format(regressor.score(X_test, y_test)))

print("\n\n\nPREDICTION")
print("-------------------------------------------------------------------------------------------------------------------")
y_pred = regressor.predict(X_test)

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\n")
print(results)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
var = np.var(y_pred)
bias = mse - var
print("\nMean absolute error: {:.2f}".format(mae))
print("\nMean squared error: {:.2f}".format(mse))
print("\nRoot mean squared error: {:.2f}".format(rmse))
print("\nVariance: {:.2f}".format(var))
print("\nBias: {:.2f}".format(bias))
x_plot = np.sort(X_test.iloc[:, 1])
y_plot = np.sort(y_pred)
plt.scatter(x_plot, np.sort(y_test),color='g')
plt.scatter(x_plot, y_plot,color='b')
plt.plot(x_plot, y_plot,color='r')
plt.legend(["Valor actual de y" , "Predicción"])
plt.title("Valor actual de y contra predicción")
plt.show()
print("-------------------------------------------------------------------------------------------------------------------")
