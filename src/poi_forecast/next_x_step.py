#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from grade_trend.def_fun import test_stationarity, acf_pacf, test_seasonal_decompose, test_arima

print 'start>>'

df=pd.read_csv('D:\\Documents\\sendi\poi_forecast\\region_popul_static.csv'
               , parse_dates=True, index_col='time')

df=df.dropna(how='any')


c = ['体育中心时尚天河', '体育中心内场', '体育中心外场']
col_name = c[1]
ts_a = df.loc['2015-11-16':'2015-11-22',[col_name]].dropna(how='any')
ts_a = ts_a[col_name]

x_step = 4*12
k = 4*8
start_index = 0+4*24*0+k
end_index = 384+4*24*0+k
ts = ts_a[ start_index : end_index ]
 
ts_diff = (ts-ts.shift()).dropna(how='any')
ts_diff2 = (ts_diff-ts_diff.shift()).dropna(how='any')
timeseries = ts_diff

print 'ts.index[-1]:',ts.index[-1]
print 'ts.ix[-1]:',ts.ix[-1] 

#plt.figure(1)
#plt.plot(ts,label='ts')
#plt.show()
#print timeseries
#test_stationarity(timeseries,7)
#acf_pacf(timeseries,30)

from statsmodels.tsa.arima_model import ARIMA


#一阶差分模型
def arima_d1(ts):
    p=0;q=0;min_rss=99999999.9
    for i in range(0,1):
        for j in range(10,20):
            try:
                model = ARIMA(ts, (i,1,j), freq='15min')
                results = model.fit(disp=-1)
                if sum((results.fittedvalues-ts_diff)**2)<min_rss:   # 0阶：ts  1阶：ts_diff
                    min_rss = sum((results.fittedvalues-ts_diff)**2)     # 0阶：ts  1阶：ts_diff
                    p=i;q=j;
            except:
                pass
    model = ARIMA(ts, order=(p, 1, q), freq='15min') 
    results = model.fit(disp=-1) 
    dr = pd.date_range(ts.index[-1],periods=x_step+1,freq='15min')
    #print dr[0],str(dr[-1])
    predictions_ARIMA_diff = results.predict(start=str(dr[0]), end=str(dr[-1]))
    #print 'predictions_ARIMA_diff:\n',predictions_ARIMA_diff
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    #print 'predictions_ARIMA_diff_cumsum:\n',predictions_ARIMA_diff_cumsum
    predictions_ARIMA = pd.Series(ts.ix[-1], index=dr)
    #print 'predictions_ARIMA:\n',predictions_ARIMA
    predictions_ARIMA = predictions_ARIMA.add(predictions_ARIMA_diff_cumsum,fill_value=0) 
    #print 'predictions_ARIMA add diff:\n',predictions_ARIMA
    forecast_ARIMA, stderr_ARIMA, conf_int = results.forecast(steps=x_step)
    forecast_ARIMA = pd.Series(forecast_ARIMA, index=dr[1:])
    #print 'forecast_ARIMA:\n',forecast_ARIMA
    #print 'stderr_ARIMA:\n',stderr_ARIMA
    return predictions_ARIMA, forecast_ARIMA

'''
predictions_ARIMA = pd.Series(); forecast_ARIMA = pd.Series()
for i in range(0,10):
    print i
    if(end_index + x_step*i > len(ts_a)) : break
    p, f = arima_d1(ts_a[ start_index + x_step*i : end_index + x_step*i ])
    predictions_ARIMA = predictions_ARIMA.append(p[1:])
    forecast_ARIMA = forecast_ARIMA.append(f)

#predictions_ARIMA, forecast_ARIMA = arima_d1(ts)
plt.figure(3)
plt.plot(ts_a[ts.index[-1]:], marker='o', linestyle='-', label='ts') 
plt.plot(predictions_ARIMA, 'x',color='r', label='predictions') 
plt.plot(forecast_ARIMA, '+', color='g', label='forecast') 
plt.legend(loc='best')
plt.title('d=1')
plt.show()
'''


#0阶差分模型
def arima_d0(ts):
    p=0;q=0;min_rss=99999999.9
    for i in range(2,3):
        for j in range(10,11):
            try:
                model = ARIMA(ts, (i,0,j), freq='15min')
                results = model.fit(disp=-1)
                if sum((results.fittedvalues-ts)**2)<min_rss:   # 0阶：ts  1阶：ts_diff
                    min_rss = sum((results.fittedvalues-ts)**2)     # 0阶：ts  1阶：ts_diff
                    p=i;q=j;
            except:
                pass
    model = ARIMA(ts, order=(p, 0, q), freq='15min') 
    results = model.fit(disp=-1) 
    dr = pd.date_range(ts.index[-1],periods=x_step+1,freq='15min')
    #print dr[0],str(dr[-1])
    predictions_ARIMA = results.predict(start=str(dr[0]), end=str(dr[-1]))
    forecast_ARIMA, stderr_ARIMA, conf_int = results.forecast(steps=x_step)
    forecast_ARIMA = pd.Series(forecast_ARIMA, index=dr[1:])
    return predictions_ARIMA, forecast_ARIMA


predictions_ARIMA = pd.Series(); forecast_ARIMA = pd.Series()
for i in range(0,10):
    print i
    if(end_index + x_step*i > len(ts_a)) : break
    p, f = arima_d0(ts_a[ start_index + x_step*i : end_index + x_step*i ])
    predictions_ARIMA = predictions_ARIMA.append(p[1:])
    forecast_ARIMA = forecast_ARIMA.append(f)
#predictions_ARIMA, forecast_ARIMA = arima_d1(ts)    
plt.figure(4)
plt.plot(ts_a[ts.index[-1]:], marker='o', linestyle='-', label='ts') 
plt.plot(predictions_ARIMA, 'x', color='r', label='predictions')
plt.plot(forecast_ARIMA, '+', color='g', label='forecast') 
plt.legend(loc='best')
plt.title('d=0')
plt.show()


print 'finished'
