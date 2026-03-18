import pytest

from hbayesdm.models import ts_akam_par6


def test_ts_akam_par6():
    _ = ts_akam_par6(
        data="example", niter=10, nwarmup=5, nchain=1, ncore=1)


if __name__ == '__main__':
    pytest.main()
