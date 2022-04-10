from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt
import datetime

class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.ewa_closing = self.datas[0].close
        self.ewc_closing = self.datas[1].close
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None
  

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close EWA, %.2f | Close EWC, %.2f' % (self.ewa_closing[0],self.ewc_closing[0]) )

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.ewa_closing[0] < self.ewa_closing[-1]:
                    # current close less than previous close

                    if self.ewa_closing[-1] < self.ewa_closing[-2]:
                        # previous close less than the previous close

                        # BUY, BUY, BUY!!! (with default parameters)
                        self.log('BUY CREATE, %.2f' % self.ewa_closing[0])

                        # Keep track of the created order to avoid a 2nd order
                        self.order = self.buy()

        else:

            # Already in the market ... we might sell
            if len(self) >= (self.bar_executed + 5):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.ewa_closing[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    #cerebro.broker.setcommission(commission=0.001)
    
    ewa =  bt.feeds.GenericCSVData(
        dataname='Stock_data/EWA.csv',
        dtformat=('%Y-%m-%d')
    )

    ewc =  bt.feeds.GenericCSVData(
        dataname='Stock_data/EWC.csv',
        dtformat=('%Y-%m-%d')
    )

    # Add the Data Feed to Cerebro
    cerebro.adddata(ewa) 
    cerebro.adddata(ewc)

    #Add Strat
    cerebro.addstrategy(TestStrategy)
    cerebro.broker.setcash(100000.0)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()
    cerebro.plot()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())