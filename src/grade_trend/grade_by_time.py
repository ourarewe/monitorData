#coding:utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from statsmodels.tsa.stattools import adfuller
from grade_trend.def_fun import test_stationarity, acf_pacf, test_seasonal_decompose, test_arima
from cProfile import label

# 6948 17994 16192 15674 15201 11331  15042 8846 12332 14894
df=pd.read_csv('D:\\Documents\\sendi\\monitorData\\rrd_host_14894.csv')

t=-1
cnt=0
scr=[]
score_py=[]
host=[]
g=0.0
x=[]
x_time=[]

for i in range(df.shape[0]):
    v=df['value'][i];th=df['tholdup'][i];l=df['warninglevel'];s=df['score'][i]
    '''
    if v>th:
        s=60.0-10.0*(4*l)-20.0*(v-th)/float(100-th)
    else:
        s=60.0+40.0*(th-v)/float(th)
    score_py.append(s)
    '''
    if t==df['t'][i]:cnt=cnt+1;g=g+s
    elif t==-1:cnt=1;t=df['t'][i];g=s
    else: 
        scr.append(g/float(cnt))
        host.append(t);cnt=1
        x.append(time.strftime("%m-%d %H",time.strptime(str(t),"%m%d%H")))
        x_time.append(pd.datetime.strptime(str(t), '%m%d%H'))
        t=df['t'][i]
        g=s
scr.append(g/float(cnt))
host.append(t)
x.append(time.strftime("%m-%d %H",time.strptime(str(t),"%m%d%H")))
x_time.append(pd.datetime.strptime(str(t), '%m%d%H'))

#df['score_py']=pd.Series(score_py)
#df.to_csv('D:\\Documents\\sendi\monitorData\\rrd_host_object_grade.csv',index=False)

    
#print scr,'\n',host,'\n',len(scr),len(host),host[3]==host[4]

scr_series=pd.Series(scr,index=x)
w = 3
scr_ma=scr_series.rolling(window=w,center=False).mean()
scr_ewma=scr_series.ewm(ignore_na=False,span=w,min_periods=0,adjust=True).mean()

df_compare=pd.DataFrame({'scr':scr,'scr_ma':scr_ma,'scr_ewma':scr_ewma})
df_compare.to_csv('D:\\Documents\\sendi\monitorData\\rrd_scr_compare.csv',index=False)

print scr,'\n',len(scr)
#print x,'\n',len(x)
#print x_time

x_ticks=range(0,len(x)+1,8)
x_label=[]
for i in x_ticks:
    x_label.append(x[i])


ax=plt.subplot(111)
plt.plot(range(len(x)), scr)
plt.title('hsot_id=6948')
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_label,rotation=15)
plt.show()
'''
ax=plt.subplot(211)
plt.plot(range(len(x)), scr, color = 'r', label='score')
pd.rolling_mean(pd.Series(scr,index=x),3).plot()
scr_ma.plot(label='ma')
scr_ewma.plot(color='g', label='ewma')
plt.legend(loc='best')
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_label,rotation=15)
plt.grid()
plt.subplot(212)
(scr_series-scr_ewma.shift()).plot()
plt.show()
'''
    
#pd.Series(scr,index=x).rolling(window=20,center=False).corr().plot(color='g')
#plt.show()


#test_stationarity(scr_series, 3)
#acf_pacf(scr_series, 20)
ts = pd.Series(scr,index=x_time)
#test_seasonal_decompose(ts)
#ts_log = np.log(ts)
#ts_predict = test_arima(ts, (1,0,1))




