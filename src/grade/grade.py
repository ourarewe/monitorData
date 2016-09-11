import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print "start>>"

df_rrd=pd.read_csv('D:\\Documents\\sendi\monitorData\\rrd_period_thold.csv')


rrd_shape = df_rrd.shape
score=[]

for i in range(rrd_shape[0]):
#for i in [0,1]:
    cnt=0
    s_item=0.0
    for j in [2,7,12,17,22]:    
        if df_rrd.iloc[i,j]!='\N':
            s_thold=100
            cnt=cnt+1
            for k in [1,2,3,4]:
                if float(df_rrd.iloc[i,j])>float(df_rrd.iloc[i,j+k]):
                    s_thold=s_thold-25
                else:
                    s_thold=s_thold-25.0*float(df_rrd.iloc[i,j])/float(df_rrd.iloc[i,j+k])
                #print "s_thold=",s_thold    
            s_item=s_item+s_thold
    score.append(s_item/float(cnt))
    print i/float(rrd_shape[0])*100.0,"%"


df_rrd['score']=pd.Series(score)
df_rrd.to_csv('D:\\Documents\\sendi\monitorData\\rrd_grade2.csv',index=False)


df_grade_host=pd.DataFrame(df_rrd.values[:,0:2])
df_grade_host['score']=pd.Series(df_rrd.values[:,27])
df_grade_host.to_csv('D:\\Documents\\sendi\monitorData\\rrd_grade_host2.csv',index=False)


print "end<<"

