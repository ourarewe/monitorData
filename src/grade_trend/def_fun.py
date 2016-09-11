from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def test_stationarity(timeseries,w): 
    #Determing rolling statistics 
    rolmean = timeseries.rolling(window=w,center=False).mean()
    rolstd = timeseries.rolling(window=w,center=False).std()
    #Plot rolling statistics: 
    plt.subplot(211)
    plt.plot(timeseries, color='blue',label='Original') 
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.legend(loc='best')
    plt.title('Original & Rolling Mean') 
    plt.subplot(212) 
    plt.plot(rolstd, color='black', label = 'Rolling Std') 
    plt.legend(loc='best') 
    plt.title('Standard Deviation') 
    plt.show() 
    #Perform Dickey-Fuller test: 
    print 'Results of Dickey-Fuller Test:' 
    dftest = adfuller(timeseries, autolag='AIC') 
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used']) 
    for key,value in dftest[4].items(): 
        dfoutput['Critical Value (%s)'%key] = value 
    print dfoutput
    
    
def acf_pacf(timeseries,n):
    from statsmodels.tsa.stattools import acf, pacf
    lag_acf = acf(timeseries, nlags=n) 
    lag_pacf = pacf(timeseries, nlags=n, method='ols')
    #Plot ACF: 
    plt.subplot(121)
    plt.bar(range(0,len(lag_acf)), lag_acf, color='g')
    #plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray') 
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray') 
    plt.axhline(y=1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray') 
    plt.title('Autocorrelation Function')
    #Plot PACF: 
    plt.subplot(122) 
    plt.bar(range(0,len(lag_pacf)), lag_pacf, color='g')
    #plt.plot(lag_pacf) 
    plt.axhline(y=0,linestyle='--',color='gray') 
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray') 
    plt.axhline(y=1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray') 
    plt.title('Partial Autocorrelation Function') 
    plt.tight_layout()
    plt.show()


def test_seasonal_decompose(timeseries):
    from statsmodels.tsa.seasonal import seasonal_decompose 
    decomposition = seasonal_decompose(timeseries) 
    trend = decomposition.trend 
    seasonal = decomposition.seasonal 
    residual = decomposition.resid 
    plt.subplot(411) 
    plt.plot(timeseries, label='Original') 
    plt.legend(loc='best') 
    plt.subplot(412) 
    plt.plot(trend, label='Trend') 
    plt.legend(loc='best') 
    plt.subplot(413) 
    plt.plot(seasonal,label='Seasonality') 
    plt.legend(loc='best') 
    plt.subplot(414) 
    plt.plot(residual, label='Residuals') 
    plt.legend(loc='best') 
    plt.tight_layout()
    plt.show()
    ts_log_decompose = residual 
    ts_log_decompose.dropna(inplace=True) 
    test_stationarity(ts_log_decompose)


def test_arima(timeseries, o):
    from statsmodels.tsa.arima_model import ARIMA
    o1 = (o[0],o[1],0); o2 = (0,o[1],o[2])
    model = ARIMA(timeseries, order=o1) 
    results_AR = model.fit(disp=-1) 
    plt.plot(timeseries, label='ts') 
    plt.plot(results_AR.fittedvalues, color='red') 
    plt.title('AR_RSS: %.4f'% sum((results_AR.fittedvalues-timeseries)**2))
    plt.show()
    model = ARIMA(timeseries, order=o2) 
    results_MA = model.fit(disp=-1) 
    plt.plot(timeseries, label='ts') 
    plt.plot(results_MA.fittedvalues, color='red') 
    plt.title('MA_RSS: %.4f'% sum((results_MA.fittedvalues-timeseries)**2))
    plt.show()    
    model = ARIMA(timeseries, order=o) 
    results_ARIMA = model.fit(disp=-1) 
    plt.plot(timeseries, label='ts') 
    plt.plot(results_ARIMA.fittedvalues, color='red') 
    plt.title('ARIMA_RSS: %.4f'% sum((results_ARIMA.fittedvalues-timeseries)**2))
    plt.show()    
    return results_ARIMA.fittedvalues
    #print 'results_ARIMA:\n',results_ARIMA
    #print 'results_ARIMA.fittedvalues:\n',results_ARIMA.fittedvalues


def m_and_v(df):
    m = df.shape[0]
    n = df.shape[1]
    df_mean = [];df_var = []
    for i in range(m):
        s=0.0;cnt=0;
        for j in range(1,n):
            if df.iloc[i,j]!=0:
                s=s+df.iloc[i,j]
                cnt=cnt+1
        df_mean.append(s/cnt)    
        print i
    for i in range(m):
        s=0.0;cnt=0;
        for j in range(1,n):
            if df.iloc[i,j]!=0:
                s=s+(df.iloc[i,j]-df_mean[i])*(df.iloc[i,j]-df_mean[i])
                cnt=cnt+1
        df_var.append(s/cnt)    
        print i    
    df['mean']=pd.Series(df_mean)
    df['var']=pd.Series(df_var)
    #df.to_csv('D:\\Documents\\sendi\monitorData\\rrd_host_time_avg.csv',index=False)   
    return df_mean, df_var

    
