#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import pandas as pd
import time
from statsmodels.tsa.stattools import adfuller
from grade_trend.def_fun import test_stationarity, acf_pacf, test_seasonal_decompose, test_arima

print 'start>>'
# poi_forecast2 region_popul_static
#dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %h:%m:%s')
df=pd.read_csv('D:\\Documents\\sendi\poi_forecast\\region_popul_static.csv'
               , parse_dates=True, index_col='time')

df=df.dropna(how='any')

'''
plt.figure(1)
df['体育中心时尚天河'].plot(color='r', label='1')
plt.figure(2)
df['体育中心内场'].plot(color='b', label='in')
plt.figure(3)
df['体育中心外场'].plot(color='g', label='out')
#plt.figure(4)
#plt.plot(df.loc['2016-06-13':'2016-06-20',['体育中心时尚天河']])
#plt.figure(5)
#plt.plot(df.loc['2016-05-16':'2016-05-23',['体育中心内场']])
#plt.figure(6)
#plt.plot(df.loc['2016-06-11':'2016-06-17',['体育中心外场']])
#plt.legend(loc='best') 
#plt.title('poi_forecast') 
#plt.figure(7)
#plt.plot(df.loc['2016-07',['体育中心时尚天河']])
plt.show()
'''

'''
c = ['体育中心时尚天河', '体育中心内场', '体育中心外场']
col_name = c[1]
ts_a = df.loc['2015-11-16':'2015-11-23',[col_name]].dropna(how='any')
ts_a = ts_a[col_name]
ts_a.to_csv('D:\\Documents\\sendi\poi_forecast\\region_popul_static.xls')

k = 22
start_index = 0+4*24*1+k*4
end_index = 4*24*4+4*24*1+k*4
ts = ts_a[ 0 :  ]
ts_diff = (ts-ts.shift()).dropna(how='any')
ts_diff2 = (ts_diff-ts_diff.shift()).dropna(how='any')
timeseries = ts
#timeseries = ts_diff

plt.figure(1)
plt.plot(ts,label='ts')
plt.plot(ts_diff,label='ts_diff')
plt.legend()
#plt.show()
#print timeseries
#test_stationarity(timeseries,7)
#acf_pacf(timeseries,200)

print ts.index[-1]
print ts.ix[-1] 
'''

'''
from statsmodels.tsa.arima_model import ARIMA
#0阶差分模型
p=0;q=0;min_rss=99999999.9
for i in range(3,6):
    for j in range(4,8):
        print i,j
        try:
            model = ARIMA(ts, (i,0,j), freq='15min')
            results = model.fit(disp=-1)
            if sum((results.fittedvalues-ts)**2)<min_rss:
                rss = sum((results.fittedvalues-ts)**2)
                p=i;q=j;
        except:
            pass
print p,q

model = ARIMA(ts, order=(p, 0, q), freq='15min') 
results = model.fit(disp=-1) 
plt.figure(20)
plt.subplot(211)
plt.plot(ts, label='ts') 
plt.plot(results.fittedvalues, color='red', label='forecast') 
plt.legend(loc='best')
plt.title('RSS: %.4f'% sum((results.fittedvalues-ts)**2))
plt.subplot(212)
plt.plot(results.fittedvalues-ts)
plt.title('diff')


#1阶差分模型
p=0;q=0;min_rss=99999999.9
for i in range(3,6):
    for j in range(4,8):
        print i,j
        try:
            model = ARIMA(ts, (i,1,j), freq='15min')
            results = model.fit(disp=-1)
            if sum((results.fittedvalues-ts_diff)**2)<min_rss:
                rss = sum((results.fittedvalues-ts_diff)**2)
                p=i;q=j;
        except:
            pass
print p,q

model = ARIMA(ts, order=(p, 1, q), freq='15min') 
results = model.fit(disp=-1)
predictions_ARIMA_diff = results.fittedvalues
res_diff_cumsum = predictions_ARIMA_diff.cumsum()
res = pd.Series(ts.ix[0], index=ts.index)
res = res.add(res_diff_cumsum,fill_value=0)
plt.figure(21)
plt.subplot(211)
plt.plot(ts, label='ts') 
plt.plot(res, color='red', label='forecast') 
plt.legend(loc='best')
plt.title('RSS: %.4f'% sum((res-ts)**2))
plt.subplot(212)
plt.plot(res-ts)
plt.title('diff')
'''

'''
predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum() 
predictions_ARIMA = pd.Series(ts.ix[0], index=ts.index)
predictions_ARIMA = predictions_ARIMA.add(predictions_ARIMA_diff_cumsum,fill_value=0) 
plt.figure(3)
plt.subplot(211)
plt.plot(ts, label='ts') 
plt.plot(predictions_ARIMA, label='predictions') 
plt.legend(loc='best')
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
plt.subplot(212)
plt.plot(predictions_ARIMA-ts)
'''

'''
et = '2016-07-15 23:45'
predictions_ARIMA_diff = results.predict(start=ts.index[-1], end=et)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum() 
predictions_ARIMA = pd.Series(ts.ix[-1], index=pd.date_range(ts.index[-2],et,freq='15min'))
predictions_ARIMA = predictions_ARIMA.add(predictions_ARIMA_diff_cumsum,fill_value=0) 
plt.figure(3)
plt.subplot(211)
plt.plot(ts, label='ts') 
plt.plot(predictions_ARIMA, label='predictions') 
plt.legend(loc='best')
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
plt.subplot(212)
plt.plot(ts_a, label='ts_a')
'''

#plt.show()

s = '2016-08-29 17:00'
ts = df.loc['2016-07-25 17:15':'2016-07-25 19:15',['体育中心时尚天河']]
print ts
m = 9
pred_df = pd.DataFrame(zeros((m,3)), index=pd.date_range(s, periods=m, freq='15min'))
arr = ts.values
print arr
print zeros((3,1))
temp_values = arr[0,0]
for i in range(9):
    try:
        pred_df.loc[i:i+1,[0]] = arr[i,0]
        temp_values = arr[i,0]
    except:
        pred_df.loc[i:i+1,[0]] = temp_values
    
print pred_df

print 'finished'
