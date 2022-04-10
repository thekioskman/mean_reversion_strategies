from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen as johansen_test
from statsmodels.tsa.stattools import coint as cadf_test
from statsmodels.regression.linear_model import OLS as least_sqaures_regression
from statsmodels.tsa.stattools import adfuller
import numpy as np
from pykalman import KalmanFilter
import logging


#DEFINE STRATEGY CLASS
class bollinger_pair_trade(bt.Strategy):

    params = (("entryLong", -1.5), ("entryShort", 1.8),("exitScoreBuffer", 0.2))

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.ewa_closing = self.datas[0].close
        self.ewc_closing = self.datas[1].close
        self.moving_z_score = self.datas[2].moving_z_score
        self.hedge_ratio = self.datas[2].hedge_ratio
        

        self.profit = 0
        self.order_counts = 0

        #check if we are in a short position of long position with respect to ewa - ewc
        self.short = False
        self.long = False
        self.orderid = None
    
    def notify_order(self, order):
        if order.status in [bt.Order.Submitted, bt.Order.Accepted]:
            return  # Await further notifications

        if order.status == order.Completed:
            if order.isbuy():
                buytxt = 'BUY COMPLETE, %.2f | commission, %.2f' % (order.executed.price * order.executed.size, order.executed.comm)
                self.log(buytxt)
            else:
                selltxt = 'SELL COMPLETE, %.2f | commission, %.2f' % (order.executed.price * order.executed.size, order.executed.comm)
                self.log(selltxt)

        elif order.status in [order.Expired, order.Canceled, order.Margin]:
            self.log('%s ,' % order.Status[order.status])
            pass  # Simply log

        # Allow new orders
        self.orderid = None
    
    def notify_trade(self, trade):
        if trade.justopened:
            self.log('Trade Opened  - Size %.2f @Price %.2f | commision: %.2f' % (trade.size, trade.price, trade.commission))
        elif trade.isclosed:
            self.log('Trade Closed  - Profit %.2f | commision: %.2f' % (trade.pnlcomm,trade.commission))
            self.profit += trade.pnlcomm

        else:  # trade updated
            self.log('Trade Updated - Size %.2f @Price %.2f'% (trade.size, trade.price))
    
    def stop(self):
        print(self.profit)
        print(self.order_counts)


    def next(self):
        #self.log('Close EWA, %.2f | Close EWC, %.2f | Moving Z Score, %.2f | Hedge_Ratio, %.2f' % (self.ewa_closing[0],self.ewc_closing[0], self.moving_z_score[0], self.hedge_ratio[0]) )
        if self.orderid:
            return  # if an order is active, no new orders are allowed


        if  0 - self.p.exitScoreBuffer < self.moving_z_score[0] < 0 + self.p.exitScoreBuffer:
            #we sell our position
            self.close(self.datas[0])
            self.close(self.datas[1])
            self.order_counts += 2

        if self.moving_z_score[0] < self.p.entryLong:
            if self.short:
                #we are still holding a short position so we realize it
                self.close(self.datas[0])
                self.close(self.datas[1])
                self.order_counts += 2
                self.short = False
            
            #we are now in a long position
            self.long = True
            #to make a long position we buy ewa and short ewc with a ratio of hedge_ratio[0]
            cash = self.stats.broker.cash[0]
            num_shares = cash / (self.ewa_closing[0] + self.hedge_ratio[0] * self.ewc_closing[0])

            #go long in ewa
            self.buy(data=self.datas[0], size = num_shares)
            #go short in ewc
            self.sell(data=self.datas[1], size = num_shares*self.hedge_ratio[0])
            self.order_counts += 2


        elif self.moving_z_score[0] > self.p.entryShort:   
            if self.long:
                #we are still holding a short position so we realize it
                self.close(self.datas[0])
                self.close(self.datas[1])
                self.order_counts += 2
                self.long = False

            #we are not in a short position
            self.short = True
            
            cash = self.stats.broker.cash[0]
            num_shares = cash / (self.ewa_closing[0] + self.hedge_ratio[0] * self.ewc_closing[0])

            #go short in ewa
            self.sell(data=self.datas[0], size = num_shares)
            #go long in ewc
            self.buy(data=self.datas[1], size = num_shares*self.hedge_ratio[0])
            self.order_counts += 2
        else:
            #we can choose to reblanace our portfolio
            pass

class IBRK_CommissionScheme(bt.CommInfoBase):
    '''
    This is a simple fixed commission scheme
    '''
    params = (
        ('commission', 1),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
        )
    
    def _getcommission(self, size, price, pseudoexec):
        if size < 100:
            return self.p.commission
        else:
            if size * 0.01 < size * price * 0.005:
                return size * 0.01
            else:
                return size * price * 0.005
            
    
class ewa_ewc_pandas(bt.feeds.PandasData):
    '''
    The ``dataname`` parameter inherited from ``feed.DataBase`` is the pandas
    DataFrame
    '''

    lines=('moving_z_score', 'hedge_ratio',)
    params = (
        # Possible values for datetime (must always be present)
        #  None : datetime is the "index" in the Pandas Dataframe
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('datetime', None),

        # Possible values below:
        #  None : column not present
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('moving_z_score', -1),
        ('hedge_ratio', -1),
    )

#MAIN BEGIN

if __name__ == '__main__':

    ewc = pd.read_csv("Stock_data/EWC.csv")
    ewa = pd.read_csv("Stock_data/EWA.csv")

    stationary_plot = pd.DataFrame()

    #USE KALMAN FILTER TO GET HEDGE RATIO AND MOVING AVERAGE
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack([ewc["Open"], np.ones(ewc["Open"].shape)]).T[:, np.newaxis]

    kf = KalmanFilter(
        n_dim_obs=1, 
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=trans_cov
    )

    state_means, state_covs = kf.filter(ewa["Open"].values)

    slope = state_means[:,0]
    intercept = state_means[:,1]

    #DERIVE HEDGED PRICE SERIES 
    stationary_plot["Plot"] = ewa["Open"] - slope * ewc["Open"]


    #USE LOOKBACK WINDOW TO GET MOVING STANDARD DEVIATION
    lookback = 20

    moving_variance = np.zeros([len(intercept)]) 

    for i in range(lookback-1, len(moving_variance)):
        moving_var = 0
        
        for loop in range(lookback):
            moving_var += (stationary_plot["Plot"].iloc[i-loop] - intercept[i])**2 
        
        moving_var /= lookback
        moving_variance[i] = moving_var

    moving_std = np.sqrt(moving_variance)


    #GET MOVING Z SCORE
    moving_z_score = (stationary_plot["Plot"] - intercept)/moving_std


    #PUT DATA INTO PANDAS DATAFRAME AND WRAP AS DATAFEED
    df = pd.DataFrame()
    df["moving_z_score"] = moving_z_score
    df["hedge_ratio"] = slope
    df["Date"] = pd.to_datetime(ewc["Date"])
    
    #to skip past the lookback window, since the first 20 values in moving_z_score will be inf
    df = df[df["Date"] > datetime.datetime(2019,3,1)]
    df.set_index("Date", inplace = True)
    
    data = ewa_ewc_pandas(dataname=df)

    ewa =  bt.feeds.GenericCSVData(
        fromdate=datetime.datetime(2019,3,1),
        dataname='Stock_data/EWA.csv',
        dtformat=('%Y-%m-%d')
    )

    ewc =  bt.feeds.GenericCSVData(
        fromdate=datetime.datetime(2019,3,1),
        dataname='Stock_data/EWC.csv',
        dtformat=('%Y-%m-%d')
    )


    #INIT BACKTRADER CEREBRO
    cerebro = bt.Cerebro()
    
    # Add the Data Feed to Cerebro
    cerebro.adddata(ewa) 
    cerebro.adddata(ewc)
    cerebro.adddata(data)

    comminfo = IBRK_CommissionScheme()
    cerebro.broker.addcommissioninfo(comminfo)

    cerebro.addstrategy(bollinger_pair_trade)
    start_cash = 20000
    cerebro.broker.setcash(start_cash)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    #cerebro.plot(volume=False)
    print(((cerebro.broker.getvalue() - start_cash)/ start_cash ))
    
    

