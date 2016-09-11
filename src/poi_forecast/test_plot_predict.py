#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as sgt
from grade_trend.def_fun import test_stationarity, acf_pacf, test_seasonal_decompose, test_arima

print 'start>>'

df=pd.read_csv('D:\\Documents\\sendi\poi_forecast\\region_popul_static.csv'
               , parse_dates=True, index_col='time')

#df=df.dropna(how='any')
df=df.fillna(method='ffill')

c = ['体育中心时尚天河', '体育中心内场', '体育中心外场']
col_name = c[1]
ts_a = df.loc[:,[col_name]].dropna(how='any')
ts_a = ts_a[col_name]


k = 0
start_index =    0 + 4*24*1 + 4*k
end_index = 4*24*3 + 4*24*1 + 4*k
#ts = ts_a[ start_index : end_index ]
ts = ts_a.loc['2016-04':'2016-07-09']
ts_diff = (ts-ts.shift()).dropna(how='any')
#print ts.head()

#plt.subplots()
#sgt.plot_acf(ts, lags=100)
#plt.subplots()
#sgt.plot_pacf(ts, lags=40)

start_day = '2015-11-19 0' + str(k)
end_day = '2015-11-20 0' + str(k)

p=0;q=0;min_rss=99999999.9
for i in range(3,6):
    for j in range(18,24):
        try:
            model = sm.tsa.ARIMA(ts, (i,0,j))
            results = model.fit(disp=-1)
            if sum((results.fittedvalues-ts)**2)<min_rss:   # 0阶：ts  1阶：ts_diff
                min_rss = sum((results.fittedvalues-ts)**2)     # 0阶：ts  1阶：ts_diff
                p=i;q=j;
        except:
            pass
print p,q,min_rss

'''
res = sm.tsa.ARIMA(ts, (p,0,q)).fit(disp=-1)

fig, ax = plt.subplots()
ax = ts.ix[start_day:].plot(ax=ax)
fig = res.plot_predict(start_day, end_day, dynamic=True, ax=ax, plot_insample=False) #dynamic=True

plt.subplots()
plt.plot(res.fittedvalues[start_day:])
plt.plot(ts.ix[start_day:], label = 'ts')
'''

plt.show()

print 'finished'