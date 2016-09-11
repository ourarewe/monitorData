# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from numpy import arctan
import matplotlib.pyplot as plt
import matplotlib as mpl
import time


print "start>>>"

'''
df_cpu=pd.read_csv('D:\Documents\sendi\monitorData\cpu_value.csv')
df_disk=pd.read_csv('D:\Documents\sendi\monitorData\disk_value.csv')
df_mem=pd.read_csv('D:\Documents\sendi\monitorData\mem_value.csv')
df_partition=pd.read_csv('D:\Documents\sendi\monitorData\partition_value.csv')
df_swap=pd.read_csv('D:\Documents\sendi\monitorData\swap_value.csv')
#df_all=pd.read_csv('D:\\Documents\\sendi\\rrd_grade.csv')
'''
df_all=pd.read_csv('D:\\Documents\\sendi\\monitorData\\rrd_grade_host2.csv')

'''
plt.subplot(3, 2, 1)
plt.hist(df_cpu.value,100)
plt.title('cpu')
plt.subplot(3, 2, 2)
plt.hist(df_disk.value,100)
plt.title('disk')
plt.subplot(3, 2, 3)
plt.hist(df_mem.value,100)
plt.title('mem')
plt.subplot(3, 2, 4)
plt.hist(df_partition.value,100)
plt.title('partition')
plt.subplot(3, 2, 5)
plt.hist(df_swap.value,100)
plt.title('swap')
plt.subplot(3, 2, 6)
'''

'''
# 分数分布
#plt.hist(df_all.values[:,37:38],100)
#plt.hist(df_all.values[:,27:28],100)
plt.hist(df_all.values[:,1:2],100)
plt.title('grade_of_host')
plt.show()
'''

'''
df_6948=pd.read_csv('D:\\Documents\\sendi\\monitorData\\rrd_6948_cpu.csv')
#x = df_6948['cpu'].values
data1=df_6948['cpu'].values
df_6948=pd.read_csv('D:\\Documents\\sendi\\monitorData\\rrd_6948_disk.csv')
data2=df_6948.values[:,1:2]
'''

# 6948 17994 16192 15674 15201 11331  15042 8846 12332 14894
df_host=pd.read_csv('D:\\Documents\\sendi\\monitorData\\rrd_12332.csv')
#df_host.dropna(how='any')
df_host.fillna(value=0)

a=[]
for i in range(len(df_host)):
        a.append(time.strftime("%m-%d %H:%M",time.strptime(str(df_host['actiontime'][i])[4:12],"%m%d%H%M")))
df_host.index=a

x_ticks_all=range(0,len(a)+1,96)
x_label_all=[]
for i in x_ticks_all:
    x_label_all.append(a[i])

x_ticks_part=range(0,60,12)
x_label_part=[]
for i in x_ticks_part:
    x_label_part.append(a[i])


title=['cpu_usage','disk_usage','mem_usage','partition_usage','swap_usage','iobusy']
t1=title[0]
t2=title[2]
data1=df_host[t1].values
data2=df_host[t2].values



ax1=plt.subplot(2, 2, 1)
df_host[t1].plot(color = 'r')
ax1.set_xticks(x_ticks_all)
ax1.set_xticklabels(x_label_all,rotation=15)
plt.title(t1)

ax2=plt.subplot(2, 2, 2)
plt.plot(range(60), data1[0:60])
ax2.set_xticks(x_ticks_part)
ax2.set_xticklabels(x_label_part,rotation=15)

ax3=plt.subplot(2, 2, 3)
df_host[t2].plot(color = 'r')
ax3.set_xticks(x_ticks_all)
ax3.set_xticklabels(x_label_all,rotation=15)
plt.title(t2)

ax4=plt.subplot(2, 2, 4)
plt.plot(range(60), data2[0:60])
ax4.set_xticks(x_ticks_part)
ax4.set_xticklabels(x_label_part,rotation=15)

plt.show()


print "end"
