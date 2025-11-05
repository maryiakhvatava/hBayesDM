context("Test ts_akam_par4_trial")
library(hBayesDM)

test_that("Test ts_akam_par4_trial", {
  # Do not run this test on CRAN
  skip_on_cran()

  expect_output(ts_akam_par4_trial(
      data = "example", niter = 10, nwarmup = 5, nchain = 1, ncore = 1))
})
