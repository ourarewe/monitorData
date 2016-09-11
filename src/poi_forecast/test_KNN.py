#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import pandas as pd
import time
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as sgt
from grade_trend.def_fun import test_stationarity
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.kdtree import KDTree

print "start"

'''
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
print indices[-1],'\n',distances[-1]
'''

df=pd.read_csv('D:\\Documents\\sendi\poi_forecast\\region_popul_static.csv'
               , parse_dates=True, index_col='time')

df=df.fillna(method='ffill')  # value=0 method='ffill'

c = ['体育中心时尚天河', '体育中心内场', '体育中心外场']
#s = str(df[-10:].index[0])
s = '2016-08-29 17:15'
dr = pd.date_range(end=s,periods=4*24+1,freq='15min')
pred_df = pd.DataFrame(zeros((9,3)), index=pd.date_range(s, periods=9, freq='15min'))

for c_i in range(3):
    col_name = c[c_i]
    ts_a = df.loc[:,[col_name]]
    ts_temp = df.loc['2015-10-23':str(dr[0]),[col_name]]
    #ts_temp = ts_temp[col_name]
    ts = ts_temp.append(df.loc[ s, [col_name] ])

    ts_shift = ts

    for i in range(1,4*12+1):
        #print i
        ts_shift = pd.merge(ts_shift, ts_a.shift(i), how='inner', left_index=True, right_index=True)

    ts_shift = ts_shift.dropna(how='any')
    #print ts_shift[15986:15986+1]
    X = ts_shift.values

    start_time = time.clock()
    #kd-tree query----------2sec
    tree = KDTree(X)
    kn = 10  # 9 nearest points 
    dist, ind = tree.query(X[-1], k=kn)
    #在9个最近点里找后面有8个值的点
    for i in range(1,kn):
        idx = ind[i]
        idx_date = str(ts_shift.index[idx])
        dr_2hours = pd.date_range(start=idx_date,periods=4*2+1,freq='15min')
        pred_array = df.loc[str(dr_2hours[0]):str(dr_2hours[-1]),[col_name]].values
        if len(pred_array)==9: break
    #如果9个最近点后面都不够8个值，则用上个数代替缺失值
    if len(pred_array)<9:
        idx = ind[1]  #如果9个最近点都不满足要求，则取回最近的点
        idx_date = str(ts_shift.index[idx])
        dr_2hours = pd.date_range(start=idx_date,periods=4*2+1,freq='15min')
        temp_array = df.loc[str(dr_2hours[0]):str(dr_2hours[-1]),[col_name]].values
        pred_array = zeros((9,1))
        temp_values = temp_array[0,0]
        for i in range(9):
            try:
                pred_array[i,0] = temp_array[i,0]
                temp_values = temp_array[i,0]
            except:
                pred_array[i,0] = temp_values 
    end_time = time.clock()
    print "read: %f s" % (end_time - start_time)
    pred_df.loc[:,[c_i]] = pred_array
    
    dr_start = pd.date_range(end=s, periods=4*12, freq='15min')
    dr_end = pd.date_range(s, periods=4*12, freq='15min')
    ts_true = ts_a[str(dr_start[0]):str(dr_end[-1])]

    dr_start = pd.date_range(end=idx_date, periods=4*12, freq='15min')
    dr_end = pd.date_range(idx_date, periods=4*12, freq='15min')
    ts_pred = ts_a[str(dr_start[0]):str(dr_end[-1])]
    ts_pred.index = pd.date_range(str(ts_true.index[0]), periods=len(ts_pred), freq='15min')

    a = ts_a.loc[s].values
    b = ts_a.loc[idx_date].values
    ts_pred = ts_pred*a/b

    plt.figure()
    plt.plot(ts_true, label='true')
    plt.plot(ts_pred, label='predict')
    plt.axvline(x=s, linestyle='--', color='gray') 
    plt.legend(loc='best')
    plt.title(idx_date)
    '''
    #-------------------------------------------------kd_tree ball_tree  3sec
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(X) 
    distances, indices = nbrs.kneighbors(X)
    idx = indices[-1,1]
    #print indices[-1,1],'\n',distances[-1,1]
    #print ts_shift.index[indices[-1,1]]
    '''

print pred_df
print type(pred_df.values[2,2])    
plt.show()

print 'finished'

