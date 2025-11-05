import pytest

from hbayesdm.models import ts_par7_trial


def test_ts_par7_trial():
    _ = ts_par7_trial(
        data="example", niter=10, nwarmup=5, nchain=1, ncore=1)


if __name__ == '__main__':
    pytest.main()
