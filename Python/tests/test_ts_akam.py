import pytest

from hbayesdm.models import ts_akam


def test_ts_akam():
    _ = ts_akam(
        data="example", niter=10, nwarmup=5, nchain=1, ncore=1)


if __name__ == '__main__':
    pytest.main()
