#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print 'start>>'

df=pd.read_csv('D:\\Documents\\sendi\poi_forecast\\region_popul_static.csv'
               , parse_dates=True, index_col='time')

df=df.dropna(how='any')

c = ['体育中心时尚天河', '体育中心内场', '体育中心外场']
col_name = c[1]

ts=df.loc['2015-11-19':'2015-11-21',[col_name]].fillna(value=0)
ts=ts[col_name]

h1 = '00'
h2 = '01'
ts1=df.loc['2015-11-20 '+h1:'2015-11-20 '+h2,[col_name]].fillna(value=0)
ts1=ts1[col_name]
ts2=df.loc['2015-11-21 '+h1:'2015-11-21 '+h2,[col_name]].fillna(value=0)
ts2=ts2[col_name]
ts3=df.loc['2015-11-19 '+h1:'2015-11-19 '+h2,[col_name]].fillna(value=0)
ts3=ts3[col_name]


a=[1,2,3]
b=[1,2,2]

print '20&21:\n',np.corrcoef(ts1,ts2)
print '20&19:\n',np.corrcoef(ts1,ts3)

print '范数：'
print '20号：',np.linalg.norm(ts1, 2)
print '21号：',np.linalg.norm(ts2, 2)
print '22号：',np.linalg.norm(ts3, 2)

print '距离：'
print '20&21:\n',np.linalg.norm(ts1.values-ts2.values)
print '20&19:\n',np.linalg.norm(ts1.values-ts3.values)

#plt.figure(1)
#plt.plot(ts, label='19-21')
#plt.figure(2)
plt.plot(ts1, label='20')
plt.plot(ts2, label='21')
plt.plot(ts3, label='19')
plt.legend(loc='best')
plt.show()

print 'finished'
