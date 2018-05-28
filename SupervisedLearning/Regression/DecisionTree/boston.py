from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

boston = load_boston()
X_train,X_test,y_train,y_test = train_test_split(boston.data,boston.target,test_size=0.25,random_state=33)

ss = StandardScaler()
ss_y = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train,y_train)
dt_y_predict = dt.predict(X_test)

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

print('R2 Score DT: ',dt.score(X_test,y_test))
print('MSE Score DT: ',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dt_y_predict)))
print('MAE Score DT: ',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dt_y_predict)))
