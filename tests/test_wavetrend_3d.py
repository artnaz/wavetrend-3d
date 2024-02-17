import numpy as np
import polars as pl
import pytest
from pathlib import Path
from src.wavetrend_3d.wavetrend_3d import WaveTrend3D


class TestWaveTrend3D:
    @pytest.fixture(scope="class")
    def df(self):
        """Fixture to load the dataframe once for all tests in this class."""
        base_dir = Path(__file__).resolve().parent
        data_file_path = base_dir / 'data/sample_btc_data.parquet'
        assert data_file_path.exists(), 'Path with sample data not found'
        return pl.read_parquet(data_file_path)

    @pytest.fixture
    def sample_numpy_data(self, df):
        """Sample numpy array data for testing."""
        return df.select('close').to_numpy()

    @pytest.fixture
    def sample_polars_series(self, df):
        """Sample Polars Series data for testing."""
        return df['close']

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
