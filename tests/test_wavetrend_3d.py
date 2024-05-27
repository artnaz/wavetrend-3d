# test_wavetrend_3d.py

import numpy as np
import pytest

from src.wavetrend_3d.plotting import PlotWaveTrend3D
from src.wavetrend_3d.wavetrend_3d import WaveTrend3D


class TestWaveTrend3D:
    def test_initialization_with_numpy_array(self, sample_numpy_data):
        """Test initialization with a Numpy array."""
        wt3d = WaveTrend3D(sample_numpy_data)
        assert isinstance(wt3d.data, np.ndarray), "Data should be stored as a Numpy array"

    def test_initialization_with_polars_series(self, sample_polars_series):
        """Test initialization with a Polars Series."""
        wt3d = WaveTrend3D(sample_polars_series)
        assert isinstance(wt3d.data, np.ndarray), "Data should be converted and stored as a Numpy array"

    def test_initialization_with_invalid_type(self):
        """Test initialization with an invalid data type."""
        with pytest.raises(ValueError):
            WaveTrend3D("invalid_data_type")

    def test_numpy_vs_series_data_same(self, sample_numpy_data, sample_polars_series):
        wt3d_numpy = WaveTrend3D(sample_numpy_data)
        wt3d_series = WaveTrend3D(sample_polars_series)
        assert np.array_equal(wt3d_numpy.data, wt3d_series.data)

    def test_get_series_output_shapes(self, sample_numpy_data):
        """Test the shapes of the output series."""
        wt3d = WaveTrend3D(sample_numpy_data)
        series_fast, series_norm, series_slow = wt3d.get_series()
        assert series_fast.shape == sample_numpy_data.flatten().shape, "Shape of series_fast should match input data"
        assert series_norm.shape == sample_numpy_data.flatten().shape, "Shape of series_norm should match input data"
        assert series_slow.shape == sample_numpy_data.flatten().shape, "Shape of series_slow should match input data"

    def test_get_series_with_optional_parameters(self, sample_numpy_data):
        """Test `get_series` method with optional parameters."""
        pass
        # wt3d = WaveTrend3D(sample_numpy_data)
        # TODO This test assumes specific behavior/effects

    def test_content_of_output(self, sample_numpy_data):
        """Test the values that are returned from "get_series()"."""
        pass
        # wt3d = WaveTrend3D(sample_numpy_data)
        # TODO This test checks the returned values

    def test_signals(self):
        pass
        # TODO This test checks the arrays with signals


class TestPlotWaveTrend3D:
    @pytest.fixture(scope='class')
    def plot(self, df, sample_polars_series):
        wt3d = WaveTrend3D(sample_polars_series)
        wt3d.compute_series()
        wt3d.compute_signals()

        plot = PlotWaveTrend3D([.7, .2, .1], height=800)
        plot.add_candles(df, 'BTC-USDT', '1h')
        plot.add_wavetrend(wt3d, main=True, mirror=True)

        return plot

    def test_plotting_data(self, plot, monkeypatch):
        # Mock the `show` method to do nothing
        # monkeypatch.setattr("plotly.graph_objs.Figure.show", lambda x: None)
        plot.show()

        assert plot.fig is not None, "Plot figure should not be None"
        assert len(plot.fig.data) > 0, "Plot figure should contain data"

        # Check if the number of traces matches the expected number
        assert len(plot.fig.data) > 10, f"Expected more than 10 traces, got {len(plot.fig.data)}"

        # Check if titles, axis labels, or other properties are set correctly
        assert plot.fig.layout.title.text is not None, "Plot should have a title"
