from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from ast import comprehension

import backtrader as bt
import datetime


class MyStrategy(bt.Strategy):

    def notify_data(self, data, status):
        print('Data Status =>', data._getstatusname(status))
        if status == data.LIVE:
            self.data_ready = True

    def next(self):
        print(self.data.datetime.datetime(0).isoformat() , self.data.open[0])

#connecting to interative brokers
print("starting")
cerebro = bt.Cerebro()
store = bt.stores.IBStore(port=7497)
data = store.getdata(dataname='USD.JPY', sectype='CASH', exchange='IDEALPRO', timeframe=bt.TimeFrame.Seconds)
cerebro.resampledata(data, timeframe=bt.TimeFrame.Seconds, compression=15)
cerebro.broker = store.getbroker()
cerebro.addstrategy(MyStrategy)

cerebro.run()

print("done")