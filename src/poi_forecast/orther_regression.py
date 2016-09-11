#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from pytz.reference import Local
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree


df=pd.read_csv('D:\\Documents\\sendi\poi_forecast\\region_popul_static.csv'
               , parse_dates=True, index_col='time')

df=df.dropna(how='any')
c = ['体育中心时尚天河', '体育中心内场', '体育中心外场']
col_name = c[1]
ts_a = df.loc['2015-11-16':'2015-11-23',[col_name]].dropna(how='any')

k=0
start_index = 4*24*0
end_index = 4*24*4  #4*24*4=384
ts = ts_a[ start_index + k : end_index + k ]
ts_diff = (ts-ts.shift()).dropna(how='any')
ts_diff2 = (ts_diff-ts_diff.shift()).dropna(how='any')

x = pd.merge(ts, ts_diff, how='inner', left_index=True, right_index=True)
x = pd.merge(x, ts_diff2, how='inner', left_index=True, right_index=True)
x = x.values
y_temp = ts_a[ start_index + k + 3 : end_index + k + 1 ].values
y=[]
for i in y_temp:
    y.extend(i)
print x,y

ts = df.loc['2015-11-21 ',[col_name]].dropna(how='any')
ts_diff = (ts-ts.shift()).dropna(how='any')
ts_diff2 = (ts_diff-ts_diff.shift()).dropna(how='any')

x_pred = pd.merge(ts, ts_diff, how='inner', left_index=True, right_index=True)
x_pred = pd.merge(x_pred, ts_diff2, how='inner', left_index=True, right_index=True)
x_pred = x_pred.values

#clf = svm.SVR(kernel='linear', C=1e3, gamma=0.1)
#clf = KNeighborsRegressor(n_neighbors=10)
#clf = tree.DecisionTreeRegressor()
#clf = linear_model.SGDRegressor()
clf = linear_model.Lasso(alpha = 1.0)
#clf = linear_model.Ridge (alpha = .5)
#clf = linear_model.LogisticRegression()

clf.fit(x, y)
y_pred = clf.predict(x_pred)

print 'x_pred:\n',x_pred[:,[0]],y_pred
print clf.coef_

x=[]
for i in x_pred[:,[0]]:
    x.extend(i)
x = pd.Series(x, index=ts.shift(2).dropna(how='any').index)
t_delta = ts.index[1] - ts.index[0] 
dr = pd.date_range(ts.index[3], ts.index[-1]+t_delta, freq='15min')
y = pd.Series(y_pred, index=dr)
plt.plot(x, label='x')
plt.plot(y, label='y')
plt.legend(loc='best')
plt.show()

print 'finished'



