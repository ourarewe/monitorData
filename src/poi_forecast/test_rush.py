#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as sgt
from grade_trend.def_fun import test_stationarity

print 'start>>'

df=pd.read_csv('D:\\Documents\\sendi\poi_forecast\\region_popul_static.csv'
               , parse_dates=True, index_col='time')

df=df.dropna(how='any')


c = ['体育中心时尚天河', '体育中心内场', '体育中心外场']
col_name = c[1]
ts_a = df.loc[:,[col_name]].dropna(how='any')
ts_a = ts_a[col_name]

ts = pd.Series()
rush_date = ['05-03', '05-14', '05-29', '06-11', '06-14', '06-24']
for d in rush_date:
    ts = ts.append(ts_a['2016-'+d])
ts.index=pd.date_range('2016-05-01', periods=len(ts), freq='15min')

#plt.plot(ts)
#plt.show()

ts_diff = (ts-ts.shift()).dropna(how='any')

'''
test_stationarity(ts,4)
sgt.plot_acf(ts)
sgt.plot_pacf(ts, lags=40)
plt.show()
'''


p=0;q=0;min_rss=99999999.9
for i in range(2,3):
    for j in range(10,11):
        try:
            model = sm.tsa.ARIMA(ts, (i,0,j))
            results = model.fit(disp=-1)
            if sum((results.fittedvalues-ts)**2)<min_rss:   # 0阶：ts  1阶：ts_diff
                min_rss = sum((results.fittedvalues-ts)**2)     # 0阶：ts  1阶：ts_diff
                p=i;q=j;
        except:
            pass
print p,q,min_rss

res = sm.tsa.ARIMA(ts, (p,0,q)).fit(disp=-1)

fig, ax = plt.subplots()
ax = ts.ix[ts.index[-4*4]:].plot(ax=ax)
dr = pd.date_range(ts.index[-4*4], periods=4*24, freq='15min')
fig = res.plot_predict(str(dr[0]), str(dr[-1]), dynamic=True, ax=ax, plot_insample=False) #dynamic=True


plt.show()


print 'finished'

