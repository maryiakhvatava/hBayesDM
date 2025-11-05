context("Test ts_akam")
library(hBayesDM)

test_that("Test ts_akam", {
  # Do not run this test on CRAN
  skip_on_cran()

  expect_output(ts_akam(
      data = "example", niter = 10, nwarmup = 5, nchain = 1, ncore = 1))
})
