#coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print "start"

df=pd.read_csv('D:\\Documents\\sendi\poi_forecast\\region_popul_static.csv'
               , parse_dates=True, index_col='time')

#df=df.dropna(how='any')
df = df.fillna(method='ffill')   #value=0 method='ffill'

c = ['体育中心时尚天河', '体育中心内场', '体育中心外场']
col_name = c[1]

ts_a = df.loc[:,[col_name]]

ts_temp = df.loc['2016-05-29',[col_name]]
ts_temp = ts_temp[col_name]
ts1 = ts_temp

ts2 = df.loc['2016-07',[col_name]]
#ts2.index = ts1.index
'''
ts_temp = df.loc['2016-05-01',[col_name]].fillna(value=0)
ts_temp = ts_temp[col_name]
ts = ts.append(ts_temp)
'''



#plt.plot(ts_a.loc['2016-06-16':'2016-06-19'], label='ts_a')
plt.plot(ts2, label='7-10')
#plt.legend()
#plt.axvline(x = ts1.index[4*6], linestyle='--', color='gray') 
plt.show()


s = '2016-05-01 00:00'
dr = pd.date_range(end=s,periods=4*24,freq='15min')
#print ts1.index
#print ts_a.loc[s].values

print "end"



