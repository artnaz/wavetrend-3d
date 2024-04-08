# conftest.py

import pytest
import polars as pl
from pathlib import Path


@pytest.fixture(scope="module")
def df():
    base_dir = Path(__file__).resolve().parent
    data_file_path = base_dir / 'data/sample_btc_data.parquet'
    assert data_file_path.exists(), 'Path with sample data not found'
    return pl.read_parquet(data_file_path)


@pytest.fixture(scope="module")
def sample_numpy_data(df):
    return df.select('close').to_numpy()


@pytest.fixture(scope="module")
def sample_polars_series(df):
    return df['close']
