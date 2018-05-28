from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

boston = load_boston()
X_train,X_test,y_train,y_test = train_test_split(boston.data,boston.target,test_size=0.25,random_state=33)

ss = StandardScaler()
ss_y = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
rfr_y_predict = rfr.predict(X_test)

etr = ExtraTreesRegressor()
etr.fit(X_train,y_train)
etr_y_predict = etr.predict(X_test)

gbr  = GradientBoostingRegressor()
gbr.fit(X_train,y_train)
gbr_y_predict = gbr.predict(X_test)

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

print('R2 Score RFR: ',rfr.score(X_test,y_test))
print('MSE Score RFR: ',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rfr_y_predict)))
print('MAE Score RFR: ',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rfr_y_predict)))
print('R2 Score ETR: ',etr.score(X_test,y_test))
print('MSE Score ETR: ',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(etr_y_predict)))
print('MAE Score ETR: ',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(etr_y_predict)))
print('R2 Score GBR: ',gbr.score(X_test,y_test))
print('MSE Score GBR: ',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(gbr_y_predict)))
print('MAE Score GBR: ',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(gbr_y_predict)))

print(etr.feature_importances_,boston.feature_names)