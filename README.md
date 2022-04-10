# Mean Reversion Strategies
Here I test out some of the mean reversion strategies found in algorithmic trading textbooks to see how well they work in practice now (how well I can actually execute them... lol). The Strategies here were read about in Algorithmic Trading by Ernest P. Chan - a great book but hard to follow at times. The initial roughwork and visualization of the strategy will be in jupyter notebook, then we will do a proper backtest in Backtrader (Python backtesting library)

# Background Information
To understand the demo strategies i will present, lets go over some of the core mathematical princeples that the algorithmis will use. We will use the examples from the strategies themselves to illustrate the concepts and by the end you will be able to peice together a mean reverting strategy from scatch.

## Stationarity and Mean Reversion
The core concept governing mean reversion strategies is the notion of stationarity. If you look into stationarity from a mathemicatcal perspective you may come across many complex defitions that are hard to visualize. In the simplest terms, we want a time series to be stationary in the sense that we know that the value of the time series will oscillate around the mean value in a "predicatable way". 

We wont every find time series that are perfectly stationaries in the world of securites trading, but our preference is for our time series to have a constant(enough) variance about the mean value.

![types of stationarity image](/images/medium%20article%20image.png)
For an in-depth look at stationarity concepts (also courtesy of image) - https://towardsdatascience.com/stationarity-in-time-series-analysis-90c94f27322

Notice that even if the mean value of our time series is variable, that doesnt really matter to us (since we can calculate the mean with a moving average). What is important is that the time series "always go back to the mean" in other words has a semi-constant variance. In the following, we will present a variety of ways to dertermine whether a time series is stationary or not. We will not go into exactly how these tests work, but rather how to anaylze their results. For these statistical tests we will be using the statsmodels library.

### What is a Hypothesis Test?
A lot of the following tests for both stationarity and cointegration are known as hypothesis tests. Essentially a statistical way for us to asses the probability of a certain assumption, in this case whether something is stationary or not. To be able to interpret the follwing tests you need to understand how a hypothesis test works.

1) Null Hypothsis - This is our initial assumption. What we first assume to be true: the time series is stationary
2) Challenger Hypothesis - This is the challange to our null hypothesis in this case it is the just the negation : the time series is not stationary
3) Test Statistic - A observation we can draw of the data *also called the trace statistic*
4) P-value - The probability of observing the test statistic

Usually you will know the probability distribution and be able to calcualte the the p-value for a given observation. But since we are using a module with abtracts away the manual calculation, we have to do a little interpolation to figure out what our test statistic is trying to tell us.
### Augmented Dicky Fuller Test
For this example, consider the following graphs, and thier results for the Augmented Dicky Fully test
![two non stationary plots](/images/staionary%20example.png)
Code Examples for image 1
```
from statsmodels.tsa.stattools import adfuller

adfuller(plot1)
adfuller(plot2)
```

```
Output
------------------------------
Plot 1:
(-0.439959559222557,
 0.9032101486279716,
 7,
 749,
 {'1%': -3.439110818166223,
  '5%': -2.8654065210185795,
  '10%': -2.568828945705979},
 -143.3804208871445)

------------------------------
Plot 2:
(-0.7982895406439698,
 0.8196239858542839,
 7,
 749,
 {'1%': -3.439110818166223,
  '5%': -2.8654065210185795,
  '10%': -2.568828945705979},
 513.3832739948602)
```

Ok lets break down the values returned by the function adfuller(). The first value in the tuple is the trace statistic. In the case of plot 1 is it aprox -0.43996. 
The second value in the tuple is the p-value. 
The third value is the number of lags used for the calculation - the lookback window
The fourth value is the number of observations used for the ADF regression and calculation

For our purposes, we can ignore the 2nd,3rd and 4th values. We care about the trace statistic and the critical values for our confidence interval (the tuple). They tell us the percent change that our test statistic was observed given that the time series is NOT stationary. Meaning the null hypothesis is that the time series is *NOT* stationary and the challenger hypothesis is that it is stationary. 

Therfore, what the output for plot 1 is telling us is that there is a 1% change of our trace statistic being -3.44 if the plot is not stationary. This essentially means that is our trace statistic is close to -3.44 we can be pretty certain that our time series is stationary. Ofcourse is follows that there is a 5% change of test statistic = -2.86 and our time series not being stationary.

As a rule of thumb,  anything < 5% means we can be pretty certain. Another thing you might have already noticed is that the trace statistics get smaller as our confidence of stationarity increase. Thus, *in this case*, we can interpret a smaller trace statistic as better evidence for us if we want the time series to be stationary. It natrually follows that if our test statistic = -100 then there is a <<<<< 1% chance that our time series is not stationary.


If trace stat at 1% <  trace stat at 5% < trace stat at 10% -> the smaller the value the more likely that the null hypothesis is wrong
BUT
if If trace stat at 1%  > trace stat at 5% > trace stat at 10% -> the bigger the value the more likely that the null hypothesis is wrong

By this logic, we can say that plot 2 is "more stationary" than plot 1, but neither are truly stationary by defintion.(Don't say this is real life, math people will murder you.)


Now consider a stationary plot
![A stationary plot](/images/stationary%20plot%20example.png)

Code Examples for image 1
```
from statsmodels.tsa.stattools import adfuller

adfuller(plot3)
```

```
Output
(-5.3391482290364545,
 4.540668357784293e-06,
 11,
 745,
 {'1%': -3.4391580196774494,
  '5%': -2.8654273226340554,
  '10%': -2.5688400274762397},
 -654.0838685347887)
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





