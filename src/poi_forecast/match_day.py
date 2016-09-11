#coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('D:\\Documents\\sendi\poi_forecast\\region_popul_static.csv'
               , parse_dates=True, index_col='time')

df=df.fillna(method='ffill')  # value=0 method='ffill'

c = ['体育中心时尚天河', '体育中心内场', '体育中心外场']
col_name = c[1]

ts = df.loc[:,[col_name]]

date_list = ['2015-11-21','2016-02-05','2016-02-06','2016-02-07','2016-03-12'
             ,'2016-03-17','2016-04-30','2016-05-03','2016-05-14','2016-05-29'
             ,'2016-06-11','2016-06-14','2016-06-24']

#plt.plot(ts)

for d in date_list:
    plt.figure()
    plt.plot(ts[d])
    plt.title(d)

plt.show()