#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from grade_trend.def_fun import test_stationarity, acf_pacf, test_seasonal_decompose, test_arima
from statsmodels.tsa.arima_model import ARIMA

print 'start>>'

df=pd.read_csv('D:\\Documents\\sendi\poi_forecast\\region_popul_static.csv'
               , parse_dates=True, index_col='time')

df=df.dropna(how='any')


c = ['体育中心时尚天河', '体育中心内场', '体育中心外场']
col_name = c[0]
#ts_a = df.loc['2015-11-16':'2015-11-23',[col_name]].dropna(how='any')
ts_a = df.loc['2016-07-12':'2016-07-18',[col_name]].dropna(how='any')
ts_a = ts_a[col_name]
#ts = ts_a[2:2+384]
start_index = 4*24*0
end_index = 4*24*4  #4*24*4=384
#ts_diff = (ts-ts.shift()).dropna(how='any')


#test_stationarity(ts,7)
#acf_pacf(ts,30)

'''
p=0;q=0;min_rss=99999999.9
for i in range(3,6):
    for j in range(8):
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
'''

'''
# 0阶差分模型
n = 96
pred = []
for k in range(n):
    print k
    ts = ts_a[ start_index + k : end_index + k ]
    p=0;q=0;min_rss=99999999.9
    for i in range(3,6):
        for j in range(4,8):
            try:
                model = ARIMA(ts, (i,0,j), freq='15min')
                results = model.fit(disp=-1)
                if sum((results.fittedvalues-ts)**2)<min_rss:
                    rss = sum((results.fittedvalues-ts)**2)
                    p=i;q=j;
            except:
                pass
    try:        
        model = ARIMA(ts, order=(p, 0, q), freq='15min') 
        results = model.fit(disp=-1)
        dr = pd.date_range(ts.index[-1],periods=2,freq='15min')
        res = results.predict(start=ts.index[-1], end=str(dr[-1]))
        pred.append(res.values[-1])
    except:
        pred.append(0)  # 或者上一个res
        pass

pred_se = pd.Series(pred, index=pd.date_range(ts_a[end_index : end_index+1].index[0],periods=n,freq='15min'))
print pred_se

path_name = 'D:\\Documents\\sendi\poi_forecast\\'+str(pred_se.index[0])[0:10]
pred_se.to_csv(path_name+'.csv')

plt.plot(ts_a[382:], label='ts')
plt.plot(pred_se, label='pred')
plt.savefig(path_name)
plt.show()
'''

# 一阶差分模型
def fun_d1(ts_a,n):
    pred = []
    for k in range(n):
        print k
        ts = ts_a[ start_index + k : end_index + k ]
        ts_diff = (ts-ts.shift()).dropna(how='any')
        p=0;q=0;min_rss=99999999.9
        for i in range(3,6):
            for j in range(4,8):
                try:
                    model = ARIMA(ts, (i,1,j), freq='15min')
                    results = model.fit(disp=-1)
                    if sum((results.fittedvalues-ts_diff)**2)<min_rss:
                        rss = sum((results.fittedvalues-ts_diff)**2)
                        p=i;q=j;
                except:
                    pass
        try:        
            model = ARIMA(ts, order=(p, 1, q), freq='15min') 
            results = model.fit(disp=-1)
            dr = pd.date_range(ts.index[-1],periods=2,freq='15min')
            res_diff = results.predict(start=ts.index[-1], end=str(dr[-1]))
            res_diff_cumsum = res_diff.cumsum()
            res = pd.Series(ts.ix[-1], index=dr)
            res = res.add(res_diff_cumsum,fill_value=0) 
            pred.append(res.values[-1])
        except:
            pred.append(0)  # 或者上一个res
            pass
    return pred

n = 96
pred = fun_d1(ts_a, n)
pred_se = pd.Series(pred, index=pd.date_range(ts_a[end_index : end_index+1].index[0],periods=n,freq='15min'))
print pred_se
path_name = 'D:\\Documents\\sendi\poi_forecast\\'+str(pred_se.index[0])[0:10]
#pred_se.to_csv(path_name+'.csv')
plt.plot(ts_a[382:], label='ts')
plt.plot(pred_se, label='pred')
#plt.savefig(path_name)
plt.show()


'''
plt.plot(ts, label='ts')
plt.show()

model = ARIMA(ts, order=(5, 0, 5), freq='15min') 
results = model.fit(disp=-1)
plt.plot(ts, label='ts') 
plt.plot(results.fittedvalues, color='r', label='predictions') 
plt.legend(loc='best')
plt.title('RMSE: %.4f'% np.sqrt(sum((results.fittedvalues-ts)**2)/len(ts)))
plt.show()
'''




