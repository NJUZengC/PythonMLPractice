from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

boston = load_boston()
# print(boston.DESCR)
X_train,X_test,y_train,y_test = train_test_split(boston.data,boston.target,test_size=0.25,random_state=33)
# print(type(y_train))

ss = StandardScaler()
ss_y = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)

from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor()
sgd.fit(X_train,y_train)
sgd_y_predict = sgd.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


print('Score LR: ',lr.score(X_test,y_test))
print('R2 Score LR: ',r2_score(y_test,lr_y_predict),r2_score(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
print('MSE Score LR: ',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
print('MAE Score LR: ',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))

print('Score SGD: ',sgd.score(X_test,y_test))
print('R2 Score SGD: ',r2_score(y_test,sgd_y_predict),r2_score(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgd_y_predict)))
print('MSE Score SGD: ',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgd_y_predict)))
print('MAE Score SGD: ',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgd_y_predict)))
