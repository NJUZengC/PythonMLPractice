import pandas  as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

column_names=['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape'
              ,'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli',
              'Mitoses','Class']
def get_data():


    data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)

    data = data.replace(to_replace='?',value=np.nan)
    data = data.dropna(how='any')
    return data


if __name__ == '__main__':
    data = get_data()
    X_train,X_test,y_train,y_test = train_test_split(data[column_names[0:10]],data[column_names[10]],test_size=0.25,random_state=33)
    #数据每个维度方差为1，均值为0
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    lr = LogisticRegression()
    sgdc = SGDClassifier()
    lr.fit(X_train,y_train)
    lr_y_predict = lr.predict(X_test)

    sgdc.fit(X_train,y_train)
    sgdc_y_predict = sgdc.predict(X_test)

    print('Accuracy LR : ',lr.score(X_test,y_test))
    print(classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant']))

    print('Accuracy SGD : ', sgdc.score(X_test, y_test))
    print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))

