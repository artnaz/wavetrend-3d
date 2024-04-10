# WaveTrend 3D

## Introduction
This is a Python Implementation of the ["WaveTrend 3D" oscillator developed in TradingView (2022, Justin Dehorty)](https://www.tradingview.com/script/clUzC70G-WaveTrend-3D/).
It is a great application of modern signal processing techniques to a well-known approach.
The "3D" aspect refers to three dimensions that are calculated, each possibly representing a different timeframe.
Arguably, it also refers to the three-dimensional visualisation, especially in "mirrored" form.


## Original description

Below is a copy from Justin Dehorty's original implementation:

> WaveTrend 3D (WT3D) is an alternative implementation of WaveTrend that directly addresses some of the known shortcomings of the indicator, including its unbounded extremes, susceptibility to whipsaw, and lack of insight into other timeframes.
> 
> In the canonical WT approach, an exponential moving average (EMA) for a given lookback window is used to assess the variability between price and two other EMAs relative to a second lookback window. Since the difference between the average price and its associated EMA is essentially unbounded, an arbitrary scaling factor of 0.015 is typically applied as a crude form of rescaling but still fails to capture 20-30% of values between the range of -100 to 100. Additionally, the trigger signal for the final EMA (i.e., TCI) crossover-based oscillator is a four-bar simple moving average (SMA), which further contributes to the net lag accumulated by the consecutive EMA calculations in the previous steps.
>
> The core idea behind WT3D is to replace the EMA-based crossover system with modern Digital Signal Processing techniques. By assuming that price action adheres approximately to a Gaussian distribution, it is possible to sidestep the scaling nightmare associated with unbounded price differentials of the original WaveTrend method by focusing instead on the alteration of the underlying Probability Distribution Function (PDF) of the input series. Furthermore, using a signal processing filter such as a Butterworth Filter, we can eliminate the need for consecutive exponential moving averages along with the associated lag they bring.
>
> Ideally, it is convenient to have the resulting probability distribution oscillate between the values of -1 and 1, with the zero line serving as a median. With this objective in mind, it is possible to borrow a common technique from the field of Machine Learning that uses a sigmoid-like activation function to transform our data set of interest. One such function is the hyperbolic tangent function (tanh), which is often used as an activation function in the hidden layers of neural networks due to its unique property of ensuring the values stay between -1 and 1. By taking the first-order derivative of our input series and normalizing it using the quadratic mean, the tanh function performs a high-quality redistribution of the input signal into the desired range of -1 to 1. Finally, using a dual-pole filter such as the Butterworth Filter popularized by John Ehlers, excessive market noise can be filtered out, leaving behind a crisp moving average with minimal lag.
>
> Furthermore, WT3D expands upon the original functionality of WT by providing:
> * First-class support for multi-timeframe (MTF) analysis
> * Kernel-based regression for trend reversal confirmation
> * Various options for signal smoothing and transformation
> * A unique mode for visualizing an input series as a symmetrical, three-dimensional waveform useful for pattern identification and cycle-related analysis
> 
> [source](https://www.tradingview.com/script/clUzC70G-WaveTrend-3D/)

## Usage

As this package is dedicated to WaveTrend 3D, applying it is very straightforward as you only need one input series.
Note however, that only a Polars Series or NumPy array is supported.
```python3
from wavetrend_3d.wavetrend_3d import WaveTrend3D

df = ... # Polars DataFrame with typical Open, High, Low, Close data

wt3d = WaveTrend3D(df['close'])
wt3d.compute_series()
wt3d.compute_signals()
```
The docstrings of the `compute_series()` method explain the settings that can be changed when calling the method.

Especially for this indicator, the visual aspect is very important. Plotting the indicator is non-trivial.
The Plotly library is used due to its dynamic, responsive features. This comes with some limitations for e.g.
visualising good-looking gradients, but the result is nevertheless acceptable in our opinion. Below we show
a few options how a candle-stick chart and WaveTrend 3D can be plotted:

```python
from wavetrend_3d.plotting import PlotWaveTrend3D

plot_mirror = True  # to easily switch whether to add a mirrored version

if plot_mirror:
  # set the image's height proportions for the price, regular oscillator, and mirrored oscillator
  plot = PlotWaveTrend3D([.7, .2, .1], height=800)
else:
  # set the proportions for the price and regular oscillator
  plot = PlotWaveTrend3D([.8, .2], height=800)

plot.add_candles(df, 'BTC-USDT', '1h')
plot.add_wavetrend(wt3d, main=True, mirror=plot_mirror)
plot.show()
```

The code above results in the following plot (note that by default, the regular white theme is used):
![generated](assets%2Fplotting_screenshot.jpg "WaveTrend 3D generated by library")

And for reference a screenshot from TradingView (note that the data is not identical):
![tradingview](assets%2Ftradingview_screenshot.jpg "WaveTrend 3D screenshot from TradingView")

## Notes

* Compared to the TradingView implementation, we use a different smoothing function to determine the long-term trend
  when computing divergences. Other than that, everything should be identical.
* Unit tests can be expanded. At this moment, the visualisation part and custom arguments are not tested.
* The dual pole filter (`apply_dual_pole_filter()`) has a loop that ideally should not be used within Python. We started
  on a vectorised version in the code, but didn't get the same results. Stopped getting it to work as there are no
  speed issues when using the function as it is now. But ideally, it should be resolved to be more "Pythonesque".

## Limitations

* Obviously, this is just one indicator and in our opinion should not be used in isolation for trading.
Ideally, this package should be used with another, more comprehensive library
* The TradingView implementation excels in beautiful views and customisation.
Here we do not support many of those settings (yet), e.g. emphasising a certain series or
separating the three dimensions.
