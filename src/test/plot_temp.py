import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('D:\\Documents\\sendi\\monitorData\\rrd_grade_predict.csv')

df['difference'].plot()
plt.show()


cnt=0
for i in range(df.shape[0]):
    if abs(df['difference'][i])>=10: cnt=cnt+1
print cnt,'\n',cnt/float(df.shape[0])
print abs(-1)

