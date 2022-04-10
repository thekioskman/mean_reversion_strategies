# Mean Reversion Strategies
Here I test out some of the mean reversion strategies found in algorithmic trading textbooks to see how well they work in practice now (how well I can actually execute them... lol). The Strategies here were read about in Algorithmic Trading by Ernest P. Chan - a great book but hard to follow at times. The initial roughwork and visualization of the strategy will be in jupyter notebook, then we will do a proper backtest in Backtrader (Python backtesting library)

# Background Information
To understand the demo strategies i will present, lets go over some of the core mathematical princeples that the algorithmis will use. We will use the examples from the strategies themselves to illustrate the concepts and by the end you will be able to peice together a mean reverting strategy from scatch.

## Stationarity and Mean Reversion
The core concept governing mean reversion strategies is the notion of stationarity. If you look into stationarity from a mathemicatcal perspective you may come across many complex defitions that are hard to visualize. In the simplest terms, we want a time series to be stationary in the sense that we know that the value of the time series will oscillate around the mean value in a "predicatable way". 

We wont every find time series that are perfectly stationaries in the world of securites trading, but our preference is for our time series to have a constant(enough) variance about the mean value.

![types of stationarity image](/images/medium%20article%20image.png)
For an in-depth look at stationarity concepts (also courtesy of image) - https://towardsdatascience.com/stationarity-in-time-series-analysis-90c94f27322

Notice that if the mean value of our time series is variable that doesnt really matter to us (since we can calculate the mean with a moving average). What is important is that the time series "always go back to the mean" in other words has a semi-constant variance. In the following, we will present a variety of ways to dertermine whether a time series is stationary or not. 

### Augmented Dicky Fuller Test

```
Code Example
```

### Hurst Exponent Variance Ratio

```
Code Example
```

## Cointegartion and Pair Trading with Mean Reversion

```
Code Example
```
### Cointegrated Augmented Dicky Fuller Test
```
Code Example
```


### Johansen Test

## Determining the Hedge Ratio

### Linear Regression

### Linear Regression with Moving Window

### Johansen Test (Advanced)

### Kalman Filter (Advanced)
Gives us:
1) Moving Average - Mean value
2) Moving Slope - Hedge ratio
3) Moving Standard Deviation - Entry and Exit signals



## Generating Entry and Exit Signals
This is the part where we turn the math into an actual trading strategy. We need to translate our mathematic factors into indicators for the algorithm of how many shares of which stocks to buy or short. We will look at two very simple mathemicatical principles that can be useful in this process.

### Half Life of Mean Reversion


### Standard Deviation
All of our entry and exit signals will be based on the standard deviation of the current price to the mean. This makes logical sense if we consider the fact that we have essentially just combined a bunch of time series in a partical way to generate a stationary time series. Since our time series is stationary, we "know it will revert back to its mean value". Therefore, when the standard deviation of the current price (for our generated time series) from the mean is >> 0 we should short our conbination of securities because the price of our combination is likely to go down toward the mean. When the standard deviation is << 0 we should be long on our combination because the prcie (value) or our combination of securities is likely to go back up toward the mean. 

Recall that standard deviation is:
<img src="https://render.githubusercontent.com/render/math?math=\sigma = \frac{\sum^i_N (x_i - \mu)}{N}">

The important thing to consider here is the value N, this is the number of data points we want to take into consideration when calcuating the standard devation. We can find the optimal value of N a number of ways. 
1) Using the Kalman Filter
2) Using the Half life of mean reversion
3) Optimiziting the paramater on data (Dangerous)



### Standard Deviation - Linear Scaling

### Standard Deviation - Bollinger Bands


# Where do you come in?
At this point it seems that the core principles behind mean reverting strategies all mostly the same. You might be thinking to yourself: So where does the induviduality come from, if any nerd can build a mean reverting strat, what seperates the winners from the losers? (OK first of all, bad mentality. We are all here to have fun :) ). I will take a few pages out of EP Chan's first book: Quantitative Trading, and give some insight into what to do from here on.

## Small Changes go a long way


## Its the context and situation that make a strategy unique


## Your Style, Your Strat





