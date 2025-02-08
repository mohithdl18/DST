import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

x, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared error : ", mse)
print('Lasso Coeeficients : ', lasso.coef_)

plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Lasso Regression')
plt.show()
