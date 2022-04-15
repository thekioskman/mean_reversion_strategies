- [Mean Reversion Strategies](#mean-reversion-strategies)
- [Background Information](#background-information)
  - [Stationarity and Mean Reversion](#stationarity-and-mean-reversion)
    - [What is a Hypothesis Test?](#what-is-a-hypothesis-test)
  - [Augmented Dicky Fuller Test](#augmented-dicky-fuller-test)
    - [Now consider a stationary plot](#now-consider-a-stationary-plot)
  - [Hurst Exponent & Variance Ratio Test](#hurst-exponent--variance-ratio-test)
  - [Cointegration and Pair Trading with Mean Reversion](#cointegration-and-pair-trading-with-mean-reversion)
  - [Determining the Hedge Ratio - Linear Combination](#determining-the-hedge-ratio---linear-combination)
  - [Cointegrated Augmented Dicky Fuller Test](#cointegrated-augmented-dicky-fuller-test)
  - [Linear Regression](#linear-regression)
  - [Linear Regression with Moving Window](#linear-regression-with-moving-window)
    - [A note of using Simple Moving Averages](#a-note-of-using-simple-moving-averages)
  - [Kalman Filter (Advanced)](#kalman-filter-advanced)
    - [Lets compare Slope/Intercept from Kalman filter to Rolling regression](#lets-compare-slopeintercept-from-kalman-filter-to-rolling-regression)
  - [Johansen Test (Advanced)](#johansen-test-advanced)
  - [Generating Entry and Exit Signals](#generating-entry-and-exit-signals)
  - [Half Life of Mean Reversion](#half-life-of-mean-reversion)
  - [Standard Deviation](#standard-deviation)
  - [Standard Deviation - Linear Scaling](#standard-deviation---linear-scaling)
  - [Standard Deviation - Bollinger Bands](#standard-deviation---bollinger-bands)
- [Testing with BackTrader](#testing-with-backtrader)
- [Where do you come in?](#where-do-you-come-in)
  - [Small Changes go a long way](#small-changes-go-a-long-way)
  - [Its the context and situation that make a strategy unique](#its-the-context-and-situation-that-make-a-strategy-unique)
  - [Your Style, Your Strat](#your-style-your-strat)

# Mean Reversion Strategies
Here I test out some of the mean reversion strategies found in algorithmic trading textbooks to see how well they work in practice now (how well I can actually execute them... lol). The Strategies here were read about in Algorithmic Trading by Ernest P. Chan - a great book but hard to follow at times. The initial roughwork and visualization of the strategy will be in jupyter notebook, then we will do a proper backtest in Backtrader (Python backtesting library)

# Background Information
To understand the demo strategies i will present, lets go over some of the core mathematical principles that the algorithm's will use. We will use the examples from the strategies themselves to illustrate the concepts and by the end you will be able to piece together a mean reverting strategy from scratch.

## Stationarity and Mean Reversion
The core concept governing mean reversion strategies is the notion of stationarity. If you look into stationarity from a mathemicatcal perspective you may come across many complex definitions that are hard to visualize. In the simplest terms, we want a time series to be stationary in the sense that we know that the value of the time series will oscillate around the mean value in a "predictable way". 

We wont every find time series that are perfectly stationary in the world of securities trading, but our preference is for our time series to have a constant(enough) variance about the mean value.

![types of stationarity image](/images/medium%20article%20image.png)
For an in-depth look at stationarity concepts (also courtesy of image) - https://towardsdatascience.com/stationarity-in-time-series-analysis-90c94f27322

Notice that even if the mean value of our time series is variable, that doesn't really matter to us (since we can calculate the mean with a moving average). What is important is that the time series "always go back to the mean" in other words has a semi-constant variance. In the following, we will present a variety of ways to determine whether a time series is stationary or not. We will not go into exactly how these tests work, but rather how to analyze their results. For these statistical tests we will be using the statsmodels library.

### What is a Hypothesis Test?
A lot of the following tests for both stationarity and cointegration are known as hypothesis tests. Essentially a statistical way for us to asses the probability of a certain assumption, in this case whether something is stationary or not. To be able to interpret the following tests you need to understand how a hypothesis test works.

1) Null Hypothesis - This is our initial assumption. What we first assume to be true: the time series is stationary
2) Challenger Hypothesis - This is the challenge to our null hypothesis in this case it is the just the negation : the time series is not stationary
3) Test Statistic - A observation we can draw of the data *also called the trace statistic*
4) P-value - The probability of observing the test statistic

Usually you will know the probability distribution and be able to calculate the the p-value for a given observation. But since we are using a module with abstracts away the manual calculation, we have to do a little interpolation to figure out what our test statistic is trying to tell us.


## Augmented Dicky Fuller Test
For this example, consider the following graphs, and their results for the Augmented Dicky Fully test
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

Ok lets break down the values returned by the function adfuller(). The first value in the tuple is the trace statistic. 
<br>
1) In the case of plot 1 it is approx -0.43996. 

2) The second value in the tuple is the p-value. 
3) The third value is the number of lags used for the calculation - the lookback window
4) The fourth value is the number of observations used for the ADF regression and calculation

<br>
For our purposes, we can ignore the 2nd , 3rd and 4th values. We care about the trace statistic and the critical values for our confidence interval (the tuple). They tell us the percent change that our test statistic was observed given that the time series is NOT stationary. Meaning the null hypothesis is that the time series is *NOT* stationary and the challenger hypothesis is that it is stationary. 

<br>
<br>
Therefore, what the output for plot 1 is telling us is that there is a 1% change of our trace statistic being -3.44 if the plot is not stationary. This essentially means that is our trace statistic is close to -3.44 we can be pretty certain that our time series is stationary. Of course is follows that there is a 5% change of test statistic = -2.86 and our time series not being stationary.

<br>
<br>
As a rule of thumb,  anything < 5% means we can be pretty certain. Another thing you might have already noticed is that the trace statistics get smaller as our confidence of stationarity increase. Thus, *in this case*, we can interpret a smaller trace statistic as better evidence for us if we want the time series to be stationary. It naturally follows that if our test statistic = -100 then there is a <<<<< 1% chance that our time series is not stationary.

<br>
<br>
If trace stat at 1% <  trace stat at 5% < trace stat at 10% -> the smaller the value, the more likely that the null hypothesis is wrong
<br>
*BUT*
<br>
if If trace stat at 1%  > trace stat at 5% > trace stat at 10% -> the bigger the value, the more likely that the null hypothesis is wrong

<br>
<br>
By this logic, we can say that plot 2 is "more stationary" than plot 1, but neither are truly stationary by definition. (Don't say this is real life, math people will murder you.)


### Now consider a stationary plot
![A stationary plot](/images/stationary%20plot%20example.png)

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

Note that again the smaller the value of the trace statistic, the more likely our time series is stationary. Since our trace stat of -5.34 < -3.44 it follow that there is a < 1% chance our time series is not stationary. In other words... it is stationary (lol). Another things to notice is that the second value 4.5e-06 is our p-value, which we can look at as the probability that we observed our test statistic given the null hypothesis is true. Therefore, we can roughly say there is a 0.00045% change that it is not stationary.

## Hurst Exponent & Variance Ratio Test
The Hurst Exponent is another criteria we can use to evaluate stationarity. The variance ratio test is just a hypothesis test to see how likely your calculated value of the hurst exponent is to be true since you are running the calculation on a finite dataset. (And more than likely just a subset of that finite dataset)

<br>
The hurst exponent is a pretty complicated here is an article that covers it pretty well -  https://towardsdatascience.com/introduction-to-the-hurst-exponent-with-code-in-python-4da0414ca52e, and is also where I got the code for calculating the hurst exponent from. 

<br>

```
def get_hurst_exponent(time_series, max_lag):
    '''
    Returns the Hurst Exponent of the time series
    '''
    lags = range(2, max_lag)


    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series.values[lag:], time_series.values[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]
```


The important thing to note here is the value of the lags, it indicates how far back we want to look into our data. In practice it is often the case that securities prices are not always mean reverting or always trending. Their movement tends to change over the course of time, they may have periods of mean reversion or periods of trending. Therefore, the time window (including both the starting point and the lag value) in which we apply the calculation of the hurst exponent can yield very different results. In the above code, we start from the latest value and just look back max_lag units of time. 

When evaluating the result of the hurst exponent we note that it is a value between 0 and 1:
hurst < 0.5 implies the time series is mean reverting
hurst = 0.5 implies the time series is a random walk
hurst > 0.5 implies the time series is trending


For information on the variance ratio test (as well as an overview of many of the topics covered in the first half of this README) -  https://medium.com/bluekiri/simple-stationarity-tests-on-time-series-ad227e2e6d48 which is also the site I ripped this code from.

```
import numpy as np

def variance_ratio(ts, lag = 2):
    '''
    Returns the variance ratio test result
    '''
    # make sure we are working with an array, convert if necessary
    ts = np.asarray(ts)
    
    # Apply the formula to calculate the test
    n = len(ts)
    mu  = sum(ts[1:n]-ts[:n-1])/n;
    m=(n-lag+1)*(1-lag/n);
    b=sum(np.square(ts[1:n]-ts[:n-1]-mu))/(n-1)
    t=sum(np.square(ts[lag:n]-ts[:n-lag]-lag*mu))/m
    return t/(lag*b);
```

In brief, we essentially just want to see a result >= 1 on the variance ratio test. Which implies that our time series is not a random walk with >= 90% confidence. Though the details are much more technical than I let on.

## Cointegration and Pair Trading with Mean Reversion
Cointegration, for out purposes is just the process of finding a linear combination of time series that will (linearly combine to) form a stationary (mean reverting) time series. It is rare(impossible) to find any stock or derivative's price that will be stationary for any meaningful amount of time. Therefore, we need to be able to snythesis a stationary time series using a combination of stocks, or other securities. The following tests will tell you if two or more time series do cointegrate (linearly combine to form a stationary time series) and give your their hedge ratio


<img src="https://render.githubusercontent.com/render/math?math=stationary(t) = a*timeSeries1(t) %2B b*timeSeries2(t) %2B c*timeSeries3(t) %2B... n**timeSeriesn(t)">
Note that this is for all t such that t is in the bounds of the window in the time series we are analyzing.

## Determining the Hedge Ratio - Linear Combination
First and foremost, lets discuss something that you might hear referred to as the hedge ratio. Essentially these are the values a,b,c...n in above equation that make sure time series 1 through n add together to a stationary time series. Although, we would it would be nice if a,b,c... n where all constant factors, the real world usually is not that nice. Hence it might produce better results if we treated the scaling factors as functions as well. Turning a,b,c...n into a(t),b(t),c(t)...n(t).

Most derivations of the hedge ratio treat the scaling factors as constant. The Kalman filter is one of the few advanced techniques that solves for the scaling factors as variable values.

## Cointegrated Augmented Dicky Fuller Test
The CADF test (Cointegrated Augmented Dicky Fuller Test) does not tell us the hedging ratio. Moreover, it can only tell us if a pair (only takes two inputs) of time series cointegrate. It is a simple to implement test, but not very effective, as it is only useful for testing pair trading strategies. (Strategies that use only two time series to create a stationary time series)

```
from statsmodels.tsa.stattools import coint as cadf_test

cadf_test(input_data["GLD"], input_data["USO"] )
```

```
Output
----------------------
(-2.272435700404821,
 0.3873847286679448,
 array([-3.89933891, -3.33774648, -3.0455719 ]))

```
Here we notice that our trace statistic is greater than the 99, 95 and 90% confidence interval values. A glance at the p-value (0.39) indicates that there is a ~39% change the null hypothesis is false. Which is not strong enough evidence to say that the two time series will conintegrate. This test is fast, but it really does not tell us much.


## Linear Regression
Another method we can use to get the hedge ratio between *two* stocks is to use a linear regression. The concept here is pretty simple, consider the following time series for the EWC and EWA ETF's. A quick CADF test tells us that they do conintegrate. We can verify this by plotting their time series values on a scatter plot, with EWC(t) on the x-axis, and EWA(t) on the y-axis.

![Linear Regression Demo](./images/Linear%20Regression%20DEMO.png)

In theory, the actual price point values of our time series should be close to, and hover around the value of the least squares regression we ran. (Since our time series value conintegrate). This means that we expect the price to mean revert around this regression line. Therefore, to generate a stationary series we use the following equation.



<img src="https://render.githubusercontent.com/render/math?math=EWA(t) = slope*EWC(t)">

And therefore
<img src="https://render.githubusercontent.com/render/math?math=stationary(t) = EWA(t) - slope*EWC(t)">

Now the idea here is that since our price values for EWA and EWC oscillate around the regression line EWA = slope*EWC, the value of the series EWA - slope*EWC should oscilate around 0. Essentially generating a mean revering time series. 

```
from statsmodels.regression.linear_model import OLS as least_sqaures_regression

linear_reg = least_sqaures_regression(ewc["Open"] , ewa["Open"])
results = linear_reg.fit()
slope = results.params[0] 
```
Calling results.params[0] gives us the slope value. However, you should consult the docs for the exact returns values. As your model might also spit out an intercept value.

<br>
But you might have noticed something important. Our regression line really is not that good. You can see period where many data points are concentrated above the line, and other regions where they are clustered below the line. This issue wil be reflected in our resulting "stationary" time series.

## Linear Regression with Moving Window
So the notion of a OLS Regression generating a constant hedge ratio was just a precursor to discussing the Rolling window Linear Regression(which will actually yield some decent results). The idea here is to treat the slope as a variable rather than a constant. Recall how I stated there are period where many values are above(or below) the line of best fit, and since the this regression line is fitted on ALL of the data it cannot account for regime shifts in the market and can quickly become outdated. Therefore, it makes logical sense to update out linear regression every so often. Then is pretty much was a rolling window OLS is. We are running a regression on a section of the data each time.
![Rolling Regression & Linear Regression Comparison](./images/Linear%20Regession%20Hedge%20Ratio%20Example.png)


As you can see the results of the rolling regression are much better than that of the regular regression. For the rolling regression in this example I used a window of 20 days, although it is often good to experiment around and see what works best for your purposes.

```
from statsmodels.regression.rolling import RollingOLS

rolling_regression = RollingOLS(ewc["Open"], ewa["Open"], window=20)
rolling_results = rolling_regression.fit()
slope_function_20 = rolling_results.params
slope_function_20 = slope_function_20.rename(columns={"Open" : "Slope"})
slope_function_20
```


### A note of using Simple Moving Averages


## Kalman Filter (Advanced)

Again we will not go into the specfics about how the Kalman filter works. We will just learn how to analyze the results we are getting from it. The Kalman filter can be a whole conversation in and of itself. All we need to know is how to extract the following components from the filter. There are many different packages that provide a Kalman filter. I am using the pykalman package, although it is a little bit outdated. A good exercise would be to figure out how the Kalman filter works, and trying to implement one yourself.

Gives us:
1) Moving Average - Mean value
2) Moving Slope - Hedge ratio
3) Moving Standard Deviation - Entry and Exit signals

```
#Dynamic Linear Regression using a Kalman Filter
from pykalman import KalmanFilter

"""
Utilise the Kalman Filter from the pyKalman package
to calculate the slope and intercept of the regressed
ETF prices.
"""
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
```

One important thing to note first is that the Kalman filter also gives us an intercept function. Which is the value that the time series is supposed to mean revert around. The rolling agression just assumes this value to be 0 (and constant).

### Lets compare Slope/Intercept from Kalman filter to Rolling regression

![Kalman vs Rolling Regression](/images/Linear%20Regression%20vs%20Kalman%20Filter.png)


## Johansen Test (Advanced)
In practice, we always prefer the Johansen Test over the CADF test.  The Johansen test can:
1) Tell us if multiple (two or more) time series conintegrate
2) It can also tell us the *constant* hedge ratios for all n time series we pass to it as inputs
3) Not dynamic is nature as it assumes the hedge ratios are constant

We will not go into the mathematics of how it works as it is significantly more complicated than linear regression. The key idea here however is that we can cointegrate and find the hedge ratio of more than two time series (which we could not do with our previous regression techniques).

```
DATA SPECS
--------------------
import yfinance as yf

#Get the data for EWC and EWA
ewc = yf.Ticker("EWC")
ewa = yf.Ticker("EWA")

ewc = ewc.history(start="2019-01-01" , end="2022-01-01")
ewa = ewa.history(start="2019-01-01" , end="2022-01-01")

```

```
from statsmodels.tsa.vector_ar.vecm import coint_johansen as johansen_test

input_data = pd.DataFrame()
input_data["EWA"] = ewa["Open"]
input_data["EWC"] = ewc["Open"]

print(input_data.columns)


print(input_data)

jres = johansen_test(input_data, 0 , 1)

trstat = jres.lr1                       # trace statistic
tsignf = jres.cvt                       # critical values
eigen_vec = jres.evec
eigen_vals = jres.eig                  

print("trace statistics", trstat)
print("critical values", tsignf)
print("Eigen vectors:", eigen_vec)
print("Eigen vals",eigen_vals)
print(eigen_vec[:,0])
```

```
Output
Index(['EWA', 'EWC'], dtype='object')
           EWA        EWC
0    16.816011  22.219603
1    16.994619  22.632471
2    17.208949  22.942117
3    17.467934  23.111017
4    17.735846  23.608328
..         ...        ...
752  24.663866  37.990002
753  24.990999  38.330002
754  24.990999  38.180000
755  25.040001  38.189999
756  24.760000  38.270000

[757 rows x 2 columns]
trace statistics [4.88385059 0.4410689 ]
critical values [[13.4294 15.4943 19.9349]
 [ 2.7055  3.8415  6.6349]]
Eigen vectors: [[ 1.10610232 -0.55180587]
 [-0.55167024  0.50269167]]
Eigen vals [0.0058672  0.00058403]
[ 1.10610232 -0.55167024]
```

There are a couple of important things to note about the Johasen test, and our data to understand why we get the above results.

1) The data is from 2019-2020 and as we can see from the linear regression model that there is not a good constant hedge ratio for such a long period of time
2) The Johansen test assumes the hedge ratio is constant, becuase of this it is much less flexible than the Kalman filter/Rolling regression in terms of classifying regression
3) Hedge ratios change over long periods of time
   

To these this hypothesis, we can simply shorten the time frame and see if the test gives us different results.





## Generating Entry and Exit Signals
This is the part where we turn the math into an actual trading strategy. We need to translate our mathematic factors into indicators for the algorithm of how many shares of which stocks to buy or short. We will look at two very simple mathemicatical principles that can be useful in this process.

## Half Life of Mean Reversion


## Standard Deviation
All of our entry and exit signals will be based on the standard deviation of the current price to the mean. This makes logical sense if we consider the fact that we have essentially just combined a bunch of time series in a partical way to generate a stationary time series. Since our time series is stationary, we "know it will revert back to its mean value". Therefore, when the standard deviation of the current price (for our generated time series) from the mean is >> 0 we should short our conbination of securities because the price of our combination is likely to go down toward the mean. When the standard deviation is << 0 we should be long on our combination because the prcie (value) or our combination of securities is likely to go back up toward the mean. 

Recall that standard deviation is:
<img src="https://render.githubusercontent.com/render/math?math=\sigma = \frac{\sum^i_N (x_i - \mu)}{N}">

The important thing to consider here is the value N, this is the number of data points we want to take into consideration when calcuating the standard devation. We can find the optimal value of N a number of ways. 
1) Using the Kalman Filter
2) Using the Half life of mean reversion
3) Optimiziting the paramater on data (Dangerous)


## Standard Deviation - Linear Scaling

## Standard Deviation - Bollinger Bands


# Testing with BackTrader

# Where do you come in?
At this point it seems that the core principles behind mean reverting strategies all mostly the same. You might be thinking to yourself: So where does the induviduality come from, if any nerd can build a mean reverting strat, what seperates the winners from the losers? (OK first of all, bad mentality. We are all here to have fun :) ). I will take a few pages out of EP Chan's first book: Quantitative Trading, and give some insight into what to do from here on.

## Small Changes go a long way


## Its the context and situation that make a strategy unique


## Your Style, Your Strat





