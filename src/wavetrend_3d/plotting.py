# plotting.py
# plots financial "candlesticks" and the WaveTrend 3D oscillator
# Copyright (C) 2024 Arthur Nazarian

from typing import List, Union
import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.wavetrend_3d.utilities import hex_to_rgba


def split_signal(signal: np.ndarray, fill=None, threshold=0):
    """
    Splits input array in two arrays depending on the provided parameters.
    Convenient function to pass more appropriate arrays for (filled) Plotly plots.

    Split a given signal array into two separate arrays: one containing
    positive values and another containing non-positive values, based on a
    specified threshold. The function offers different methods to handle the
    missing values in the output arrays.

    Parameters
    ----------
    signal : np.ndarray
        The input signal as a numpy array.

    fill : {None, 'bfill', 'zero'}, optional, default: None
        The method to handle missing values in the output arrays after splitting:
        - None: Replace missing values (the elements that go into the other split) with NaN.
        - 'bfill' (backwards fill): Fill the single element of a missing value prior to a signal,
          with the value of the opposite array.
          This ensures that when both signals are plotted, there is a continuous line
        - 'zero': "surround" the signal with one element of zero.

    threshold : int or float, optional, default: 0
        The value used to split the signal into positive and non-positive values.
        Values greater than the threshold are considered positive, while values
        less than or equal to the threshold are considered non-positive.

    Returns
    -------
    pos_out : np.ndarray
        The output array containing positive values based on the specified threshold
        and fill method.

    neg_out : np.ndarray
        The output array containing non-positive values based on the specified threshold
        and fill method.

    Examples
    --------
    >> import numpy as np
    >> import polars as pl
    >>
    >> x = np.array([-1, 1, 2, 1, -1, -2, 1])
    >> pos_reg, neg_reg = split_signal(x, None)
    [nan  1.  2.  1. nan nan  1.], [-1. nan nan nan -1. -2. nan]
    >> pos_bfill, neg_bfill = split_signal(x, 'bfill')
    [-1.  1.  2.  1. nan -2.  1.], [-1. nan nan  1. -1. -2. nan]
    >> pos_zero, neg_zero = split_signal(x, 'zero')
    [ 0.  1.  2.  1. 0.   0.  1.], [-1.  0. nan  0. -1. -2.  0.]
    """

    if fill is None:
        return np.where(signal > threshold, signal, np.nan), np.where(signal <= threshold, signal, np.nan)

    elif fill == 'bfill':
        df = pl.DataFrame({'signal': signal})

        out = (
            df
            .with_columns(
                pos=pl.when(pl.col('signal') > 0).then(pl.col('signal')).otherwise(None),
                neg=pl.when(pl.col('signal') <= 0).then(pl.col('signal')).otherwise(None)
            )
            .with_columns(
                pos_n1=pl.col('pos').shift(-1),
                neg_n1=pl.col('neg').shift(-1),
            )
            .with_columns(
                pos_out=(
                    pl
                    .when(pl.col('pos').is_not_null())
                    .then(pl.col('pos'))
                    .when(pl.col('pos').is_null() & pl.col('pos_n1').is_not_null())
                    .then(pl.col('neg'))
                    .otherwise(None)
                ),
                neg_out=(
                    pl
                    .when(pl.col('neg').is_not_null())
                    .then(pl.col('neg'))
                    .when(pl.col('neg').is_null() & pl.col('neg_n1').is_not_null())
                    .then(pl.col('pos'))
                    .otherwise(None)
                ),
            )
            .select('pos_out', 'neg_out').to_numpy()
        )
        return out[:, 0], out[:, 1]

    elif fill == 'zero':
        df = pl.DataFrame({'signal': signal})

        out = (
            df
            .with_columns(
                pos=pl.when(pl.col('signal') > 0).then(pl.col('signal')).otherwise(None),
                neg=pl.when(pl.col('signal') <= 0).then(pl.col('signal')).otherwise(None)
            )
            .with_columns(
                pos_p1=pl.col('pos').shift(1),
                neg_p1=pl.col('neg').shift(1),
                pos_n1=pl.col('pos').shift(-1),
                neg_n1=pl.col('neg').shift(-1),
            )
            .with_columns(
                pos_out=(
                    pl
                    .when(pl.col('pos').is_not_null())
                    .then(pl.col('pos'))
                    .when(pl.col('pos').is_null() & (pl.col('pos_p1').is_not_null() | pl.col('pos_n1').is_not_null()))
                    .then(0)
                    .otherwise(None)
                ),
                neg_out=(
                    pl
                    .when(pl.col('neg').is_not_null())
                    .then(pl.col('neg'))
                    .when(pl.col('neg').is_null() & (pl.col('neg_p1').is_not_null() | pl.col('neg_n1').is_not_null()))
                    .then(0)
                    .otherwise(None)
                ),
            )
            .select('pos_out', 'neg_out').to_numpy()
        )
        return out[:, 0], out[:, 1]

    else:
        raise ValueError("`fill` argument should be either None, 'bfill' or 'zero'.")


def add_oscillator(
        fig: go.Figure, datetime: np.ndarray, signal: np.ndarray, name: str,
        row: int = None, line_opacity: float = 1., fill_opacity: float = 1.,
        line_width: Union[float, int] = 1, colour_pos=None, colour_neg=None,
        show_legend: bool = True
) -> go.Figure:
    """
    Add an oscillator subplot to a Plotly Figure with positive and negative values and a fill until the zero line
     on x-axis. Four traces will be plotted (not two) to make the fill better-looking.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The Figure object to which the oscillator subplot will be added.
    datetime : np.ndarray
        An array of datetime values corresponding to the x-axis values.
    signal : np.ndarray
        An array of numerical values corresponding to the y-axis values. This will be split into positive and
        negative arrays to be plotted nicely.
    name : str
        The name of the oscillator, used in the legend. Also used for the `legendgroup` argument, which should
        be unique across the plot.
    row : int, optional
        The row index of the subplot in the figure (required for a subplot).
    line_opacity : float, optional, default: 1.0
        The colour opacity (`= 1 - transparency`) used to display the lines of the oscillator; float in (0.0, 1.0).
    fill_opacity : float, optional, default: 1.0
        The colour opacity (`= 1 - transparency`) used to display the fill between zero and the lines of the oscillator;
        float in (0.0, 1.0).
    line_width : float or int, optional, default: 1
        The width for the lines in the plot
    colour_pos : str, optional
        The color used to display positive values in the oscillator (default is '#009988', Google green).
    colour_neg : str, optional
        The color used to display negative values in the oscillator (default is '#CC3311', Google red).
    show_legend : bool, optional, default: True
        Whether to show the series in the legend. In that case, only the lines will be shown in the legend,
        but they all will be connected as a group to be able to select and deselect interactively.

    Returns
    -------
    plotly.graph_objects.Figure
        The updated Figure object with the added oscillator subplot.

    Notes
    -----
    The `split_signal()` function is assumed to be available in the same module, returning two arrays
    with the positive and negative values of the input signal array, respectively.
    """
    if row is None:
        raise NotImplementedError(
            'Only plotting a oscillator is not implemented, should be a subplot; thus provide `row`'
        )
    if (line_opacity < 0.) | (line_opacity > 1.) | (fill_opacity < 0.) | (fill_opacity > 1.):
        raise ValueError('The colour transparencies should be floats in the range [0.0, 1.0]')

    if (line_width < 0) | (not isinstance(line_width, (float, int))):
        raise ValueError(f'The `line_width` argument should be positive number. Received: {line_width}`')

    colour_pos = '#009988' if colour_pos is None else colour_pos  # standard green
    colour_neg = '#CC3311' if colour_neg is None else colour_neg  # standard red
    transparent_colour = 'rgba(0, 0, 0, 0)'

    signal_pos_bfill, signal_neg_bfill = split_signal(signal, 'bfill')
    signal_pos_zero, signal_neg_zero = split_signal(signal, 'zero')

    # positive fill (this could have been done in one step, but Plotly makes the fill ugly)
    fig.add_trace(go.Scatter(  # positive fill
        x=datetime, y=signal_pos_zero,
        name=f'{name} + fill', legendgroup=name, showlegend=False,
        line=dict(color=transparent_colour, width=0),  # do not show the line
        fill='tozeroy',
        fillcolor=hex_to_rgba(colour_pos, fill_opacity)
    ), row=row, col=1)
    # negative fill
    fig.add_trace(go.Scatter(
        x=datetime, y=signal_neg_zero,
        name=f'{name} - fill', legendgroup=name, showlegend=False,
        line=dict(color=transparent_colour, width=0),  # do not show the line
        fill='tozeroy',
        fillcolor=hex_to_rgba(colour_neg, fill_opacity),
    ), row=row, col=1)
    # positive line
    fig.add_trace(go.Scatter(
        x=datetime, y=signal_pos_bfill,
        name=f'{name} +', legendgroup=name, showlegend=show_legend,
        line=dict(color=hex_to_rgba(colour_pos, line_opacity), width=line_width)
    ), row=row, col=1)
    # negative line
    fig.add_trace(go.Scatter(
        x=datetime, y=signal_neg_bfill,
        name=f'{name} -', legendgroup=name, showlegend=show_legend,
        line=dict(color=hex_to_rgba(colour_neg, line_opacity), width=line_width)
    ), row=row, col=1)

    return fig


class PlotWaveTrend3D:
    def __init__(self, subplot_proportions: List[float] = None, height: int = 1200):
        self.fig = None
        self.subplot_proportions = subplot_proportions
        self.height = height

        self.colour_bull = '#009988'
        self.colour_bear = '#CC3311'
        self.num_subplots = len(subplot_proportions)

    def add_candles(self, df: pl.DataFrame, market: str = None, interval: str = None):
        self._initialize_ohlc(df, market, interval)
        self._fig_main_candles()

    def add_oscillators(self, signal_fast, signal_norm, signal_slow, main=True, mirror=False):
        """'main' is the regular WaveTrend3D oscillator, 'mirror' is the symmetrical/mirrored one."""
        if main:
            self._add_main_oscillator(signal_fast, signal_norm, signal_slow)
        if mirror:
            self._add_mirror_oscillator(signal_fast, signal_norm, signal_slow)

    def _add_main_oscillator(self, signal_fast, signal_norm, signal_slow):
        general_config = dict(fig=self.fig, datetime=self.datetime, show_legend=True, row=2)
        self.fig = add_oscillator(signal=signal_slow, name='WT3D signal (slow)',
                                  line_opacity=0.7, fill_opacity=0.1, line_width=3, **general_config)
        self.fig = add_oscillator(signal=signal_norm, name='WT3D signal (norm)',
                                  line_opacity=0.9, fill_opacity=0.1, line_width=1.5, **general_config)
        self.fig = add_oscillator(signal=signal_fast, name='WT3D signal (fast)',
                                  line_opacity=0.8, fill_opacity=0.1, line_width=0.5, **general_config)

    def _add_mirror_oscillator(self, signal_fast, signal_norm, signal_slow):
        general_config = dict(fig=self.fig, datetime=self.datetime, row=3, line_opacity=0.2, fill_opacity=0.12)
        self.fig = add_oscillator(signal=signal_slow, name='WT3D mirror (slow)', show_legend=True,
                                  line_width=1, **general_config)
        self.fig = add_oscillator(signal=-signal_slow, name='WT3D mirror (slow)', show_legend=False,
                                  line_width=1, **general_config)

        self.fig = add_oscillator(signal=signal_norm, name='WT3D mirror (norm)', show_legend=True,
                                  line_width=1, **general_config)
        self.fig = add_oscillator(signal=-signal_norm, name='WT3D mirror (norm)', show_legend=False,
                                  line_width=1, **general_config)

        self.fig = add_oscillator(signal=signal_fast, name='WT3D mirror (fast)', show_legend=True,
                                  line_width=1, **general_config)
        self.fig = add_oscillator(signal=-signal_fast, name='WT3D mirror (fast)', show_legend=False,
                                  line_width=1, **general_config)

    def _initialize_ohlc(self, df: pl.DataFrame, market=None, interval=None):
        if not isinstance(df, pl.DataFrame):
            raise ValueError('Expects a Polars DataFrame for `df`.')
        for c in ['datetime', 'open', 'high', 'low', 'close']:
            if c not in df.columns:
                raise ValueError(f'Column "{c}" not found in passed DataFrame')

        self.datetime = df['datetime']
        self.open = df['open']
        self.high = df['high']
        self.low = df['low']
        self.close = df['close']

        if market is None:
            if 'market' in df.columns:
                markets = df['market'].unique()
                if len(markets) > 1:
                    raise ValueError(f'Only one market is supported. Received {len(markets)} markets: {markets}')
                else:
                    self.market = df.select('market').item(0, 0)
            else:
                raise ValueError('Please either provide a name for `market` or include it as a column.')
        else:
            self.market = market

        if interval is None:
            if 'interval' in df.columns:
                intervals = df['interval'].unique()
                if intervals > 1:
                    raise ValueError(f'Only one interval is supported. Received {len(intervals)} markets: {intervals}')
                else:
                    self.interval = df.select('interval').item(0, 0)
            else:
                raise ValueError('Please either provide a name for `interval` or include it as a column.')
        else:
            self.interval = interval

    def _fig_main_candles(self):
        self.fig = make_subplots(
            rows=self.num_subplots, cols=1, shared_xaxes=True,
            vertical_spacing=0.005, row_heights=self.subplot_proportions
        )

        self.fig.add_trace(go.Candlestick(
            x=self.datetime,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            name='price'
        ), row=1, col=1)

    def _fig_final_layout(self):
        self.fig.update_xaxes(
            dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]),
                ),
                rangeslider=dict(visible=False),
                type='date',
            ),
            rangeslider_thickness=0.2,
            row=1, col=1
        )

        # Layout
        self.fig.update_yaxes(title_text='price', side='right', row=1, col=1)
        self.fig.update_yaxes(title_text='WT3D signal', side='right', row=2, col=1)
        self.fig.update_yaxes(title_text='WT3D mirror', side='right', row=3, col=1)

        main_title = f'Price {self.market} ({self.interval}) and "WaveTrend 3D oscillator"'
        self.fig.update_layout(
            template='plotly_white',
            title=main_title,
            showlegend=True,
            height=self.height,
            margin=go.layout.Margin(
                l=40,  # left margin
                r=40,  # right margin
                b=40,  # bottom margin
                t=100  # top margin
            )
        )

    def show(self):
        self._fig_final_layout()
        self.fig.show()
