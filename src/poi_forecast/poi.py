#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from grade_trend.def_fun import test_stationarity, acf_pacf, test_seasonal_decompose, test_arima

df=pd.read_csv('D:\\Documents\\sendi\poi_forecast\\poi_forecast_8_10.csv'
               , parse_dates=True, index_col='time')
df=df[1:70].astype(float)
df=df.dropna(how='any')

df_fore=pd.read_csv('D:\\Documents\\sendi\poi_forecast\\poi_forecast_8_10_forecast_count.csv'
               , parse_dates=True, index_col='time')
df_fore=df_fore.astype(float)
df_fore=df_fore.dropna(how='any')

'''
plt.figure(1)
df['体育中心时尚天河'].plot(color='r', label='1')
plt.figure(2)
df['体育中心内场'].plot(color='b', label='in')
plt.figure(3)
df['体育中心外场'].plot(color='g', label='out')
plt.show()
'''

c = ['体育中心时尚天河', '体育中心内场', '体育中心外场']
col_name = c[0]
ts=df.loc[:,[col_name]]
ts=ts[col_name]
ts_pred = df_fore.loc[:,[col_name]]
ts_pred = ts_pred[col_name]
plt.figure(1)
plt.subplot(211)
plt.plot(ts,label='ts')
plt.plot(ts_pred,label='ts_pred')
plt.legend(loc='best')
plt.subplot(212)
plt.plot(ts-ts_pred,label='diff')

#print ts

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts, order=(1, 1, 1), freq='15min') 
results = model.fit(disp=-1)
res_diff = results.predict(start='2016-08-10 00:45', end='2016-08-10 23:45')
res_diff_cumsum = res_diff.cumsum() 
res = pd.Series(ts.ix[0], index=ts.index)
res = res.add(res_diff_cumsum,fill_value=0) 
plt.figure(2)
plt.subplot(211)
plt.plot(ts,label='ts')
plt.plot(res,label='ts_pred')
plt.legend(loc='best')
plt.subplot(212)
plt.plot(ts-res,label='diff')

plt.show()


''' 
plt.subplot(211)
plt.plot(ts, label='ts') 
plt.plot(results.fittedvalues, color='red', label='forecast') 
plt.legend(loc='best')
plt.title('RSS: %.4f'% sum((results.fittedvalues-ts)**2))
plt.subplot(212)
plt.plot(results.fittedvalues-ts)
plt.title('diff')
plt.show()
'''

'''
print results.fittedvalues
results = ARIMA(ts[0:35], order=(4, 0, 0), freq='15min').fit(disp=-1)
res = results.predict(start='2016-08-09 7:30:02', end='2016-08-09 10:45:02')
#res = results.predict(start=0, end=0)
print res
plt.plot(ts, color='g', label='ts')
plt.plot(results.fittedvalues, color='b')
plt.plot(res, color='r')
plt.show()

rng = pd.date_range('2011-12-29 10:45:02', '2011-12-31 10:49:02',freq='15min')
ts = pd.Series(np.random.randn(len(rng)),rng)
#print ts
'''


print 'finished'

