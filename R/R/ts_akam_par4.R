#' @templateVar MODEL_FUNCTION ts_akam_par4
#' @templateVar CONTRIBUTOR \href{None}{Franziska Knolle/Maryia} <\email{franziskaknolle@@gmail.com}>
#' @templateVar TASK_NAME Two-Step Task
#' @templateVar TASK_CODE ts
#' @templateVar TASK_CITE (Daw et al., 2011)
#' @templateVar MODEL_NAME Hybrid Model for reduced task
#' @templateVar MODEL_CODE akam_par4
#' @templateVar MODEL_CITE (Akam et al., 2015)
#' @templateVar MODEL_TYPE Hierarchical
#' @templateVar DATA_COLUMNS "subjID", "level1_choice", "level2_choice", "reward"
#' @templateVar PARAMETERS \code{a} (learning rate for both stages 1 & 2), \code{beta} (inverse temperature for both stages 1 & 2), \code{pi} (perseverance), \code{w} (model-based weight)
#' @templateVar REGRESSORS "mf_RPE", "mb_RPE", "mfb_RPE"
#' @templateVar POSTPREDS "y_pred_step1", "y_pred_step2"
#' @templateVar LENGTH_DATA_COLUMNS 4
#' @templateVar DETAILS_DATA_1 \item{subjID}{A unique identifier for each subject in the data-set.}
#' @templateVar DETAILS_DATA_2 \item{level1_choice}{Choice made for Level (Stage) 1 (1: stimulus 1, 2: stimulus 2).}
#' @templateVar DETAILS_DATA_3 \item{level2_choice}{Choice made for Level (Stage) 2 (1: stimulus 3, 2: stimulus 4).\cr        Note that, in our notation, choosing stimulus 1 in Level 1 leads to stimulus 3 in Level 2 with a common (0.7 by default) transition. Similarly, choosing stimulus 2 in Level 1 leads to stimulus 4 in Level 2 with a common (0.7 by default) transition. To change this default transition probability, set the function argument `trans_prob` to your preferred value.}
#' @templateVar DETAILS_DATA_4 \item{reward}{Reward after Level 2 (0 or 1).}
#' @templateVar LENGTH_ADDITIONAL_ARGS 1
#' @templateVar ADDITIONAL_ARGS_1 \item{trans_prob}{Common state transition probability from Stage (Level) 1 to Stage (Level) 2. Defaults to 0.7.}
#'
#' @template model-documentation
#'
#' @export
#' @include hBayesDM_model.R
#' @include preprocess_funcs.R

#' @references
#' Daw, N. D., Gershman, S. J., Seymour, B., Ben Seymour, Dayan, P., & Dolan, R. J. (2011). Model-Based Influences on Humans' Choices and Striatal Prediction Errors. Neuron, 69(6), 1204-1215. https://doi.org/10.1016/j.neuron.2011.02.027
#'
#' Akam, T., Costa, R., & Dayan, P. (2015). Simple Plans or Sophisticated Habits? State, Transition and Learning Interactions in the Two-Step Task. PLoS Comput Biol, 11(12):e1004648. https://doi.org/10.1371/journal.pcbi.1004648
#'


ts_akam_par4 <- hBayesDM_model(
  task_name       = "ts",
  model_name      = "akam_par4",
  model_type      = "",
  data_columns    = c("subjID", "level1_choice", "level2_choice", "reward"),
  parameters      = list(
    "a" = c(0, 0.5, 1),
    "beta" = c(0, 1, Inf),
    "pi" = c(0, 1, 5),
    "w" = c(0, 0.5, 1)
  ),
  additional_args = list(
    'trans_prob' = 0.7
  ),
  regressors      = list(
    "mf_RPE" = 2,
    "mb_RPE" = 2,
    "mfb_RPE" = 2
  ),
  postpreds       = c("y_pred_step1", "y_pred_step2"),
  preprocess_func = ts_preprocess_func)
