#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from grade_trend.def_fun import m_and_v

'''
df=pd.read_csv('D:\\Documents\\sendi\\monitorData\\rrd_host_time.csv')
m = df.shape[0]
n = df.shape[1]
print 'm=',m,'\tn=',n
print 'range(1,n)=',range(1,n)
print df.iloc[0,65]
df_mean = [];df_var = []
for i in range(m):
    s=0.0;cnt=0;
    for j in range(1,n):
        if df.iloc[i,j]!=0:
            s=s+df.iloc[i,j]
            cnt=cnt+1
    df_mean.append(s/cnt)    
    print i
    
print df_mean,'\n'

for i in range(m):
    s=0.0;cnt=0;
    for j in range(1,n):
        if df.iloc[i,j]!=0:
            s=s+(df.iloc[i,j]-df_mean[i])*(df.iloc[i,j]-df_mean[i])
            cnt=cnt+1
    df_var.append(s/cnt)    
    print i
 
print df_var     

df['mean']=pd.Series(df_mean)
df['var']=pd.Series(df_var)
df.to_csv('D:\\Documents\\sendi\monitorData\\rrd_host_time_avg.csv',index=False)
'''

#df=pd.read_csv('D:\\Documents\\sendi\\monitorData\\rrd_host_time.csv')
#df_mean, df_var = m_and_v(df)

df=pd.read_csv('D:\\Documents\\sendi\\monitorData\\rrd_host_time_avg.csv')
plt.plot(df['var'])
plt.show()

plt.hist(df['var'],100)
plt.show()



