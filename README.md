# Mean Reversion Strategies
Here I test out some of the mean reversion strategies found in algorithmic trading textbooks to see how well they work in practice now (how well I can actually execute them... lol). The Strategies here were read about in Algorithmic Trading by Ernest P. Chan - a great book but hard to follow at times. The initial roughwork and visualization of the strategy will be in jupyter notebook, then we will do a proper backtest in Backtrader (Python backtesting library)

# Background Information

## Stationarity and Mean Reversion


### Augmented Dicky Fuller Test


### Hurst Exponent Variance Ratio



## Cointegartion and Pair Trading with Mean Reversion


### Cointegrated Augmented Dicky Fuller Test



### Johansen Test

## Determining the Hedge Ratio

### Linear Regression

### Linear Regression with Moving Window

### Johansen Test (Advanced)

### Kalman Filter (Advanced)



## Generating Entry and Exit Signals


### Standard Deviation
All of our entry and exit signals will be based on the standard deviation of the current price to the mean. This makes logical sense if we consider the fact that we have essentially just combined a bunch of time series in a partical way to generate a stationary time series. Since our time series is stationary, we "know it will revert back to its mean value". Therefore, when the standard deviation of the current price (for our generated time series) from the mean is >> 0 we should short our conbination of securities because the price of our combination is likely to go down toward the mean. When the standard deviation is << 0 we should be long on our combination because the prcie (value) or our combination of securities is likely to go back up toward the mean. 

<img src="https://render.githubusercontent.com/render/math?math=\sigma = (\sum_N (x_i - \mu))/N">


### Standard Deviation - Linear Scaling

### Standard Deviation - Bollinger Bands





