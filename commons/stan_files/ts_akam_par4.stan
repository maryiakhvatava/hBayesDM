// Reduced two-step task - Akam et al., 2015
// -----------------------------------------------------------
// Taskstructure
// - level1_choice: 1 or 2 (first-stage left/right)
// - level2_choice: 1 or 2 (only two second-stage choices in total, each left/right goes to one choice on second stage)
// - v_mf -> 4 elements: 1-2 correspond to level-1
//   action values, indices 3-4 correspond to the two second-stage
//   action values.
//


#include /pre/license.stan

data {
  int<lower=1> N;                          // number of subjects
  int<lower=1> T;                          // maximum number of trials
  int<lower=1, upper=T> Tsubj[N];          // number of trials for each subject
  int<lower=1, upper=2> level1_choice[N,T]; // 1:left, 2:right  (first-stage)
  int<lower=1, upper=2> level2_choice[N,T]; // 1 or 2 (I CHANGED THIS: now reduced second-stage choices)
  int<lower=0, upper=1> reward[N,T];      // reward observed at level 2 (0/1)
  real<lower=0, upper=1> trans_prob;      // transition probability (prob of common transition)
}

transformed data {
  // empty in original, nothing transformed required
}

parameters {
  // Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[4] mu_pr;
  vector<lower=0>[4] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] a_pr;
  vector[N] beta_pr;
  vector[N] pi_pr;
  vector[N] w_pr;
}
transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0,upper=1>[N] a;
  vector<lower=0>[N]         beta;
  vector<lower=0,upper=5>[N] pi;
  vector<lower=0,upper=1>[N] w;

  for (i in 1:N) {
      a[i]     = Phi_approx( mu_pr[1] + sigma[1] * a_pr[i] );
      beta[i]  = exp( mu_pr[2] + sigma[2] * beta_pr[i] );
      pi[i]    = Phi_approx( mu_pr[3] + sigma[3] * pi_pr[i] ) * 5;
      w[i]     = Phi_approx( mu_pr[4] + sigma[4] * w_pr[i] );
  }
}

model {
  // Hyperparameters
  mu_pr  ~ normal(0, 1);
  sigma ~ normal(0, 0.2);

  // individual parameters
  a_pr     ~ normal(0, 1);
  beta_pr  ~ normal(0, 1);
  pi_pr    ~ normal(0, 1);
  w_pr     ~ normal(0, 1);

  // Main likelihood: loop over subjects and trials
  for (i in 1:N) {
    // Local value vectors for each subject's run -- reduced: 4 elements only
    vector[2] v_mb;      // model-based values for level 1 (two first-stage stimuli)
    vector[4] v_mf;      // model-free values: 1-2 -> level1; 3-4 -> level2 (reduced)
    vector[2] v_hybrid;  // hybrid values for level 1 (weighted by w)

    // helper variables
    real level1_prob_choice2;
    real level2_prob_choice2;
    int level1_choice_01;
    int level2_choice_01;

    // initialize values to 0 at the start of each subject
    v_mb      = rep_vector(0.0, 2); //Model-based value of choice
    v_mf      = rep_vector(0.0, 4); //Model-free value of choice
    v_hybrid  = rep_vector(0.0, 2);

    for (t in 1:Tsubj[i]) {
      // Model-based value computation -> I CHANGED THIS
      // choice 1: common → state A (v_mf[3]), rare → state B (v_mf[4])
      v_mb[1] = trans_prob * v_mf[3] + (1 - trans_prob) * v_mf[4];
      // choice 2: common → state B (v_mf[4]), rare → state A (v_mf[3])
      v_mb[2] = (1 - trans_prob) * v_mf[3] + trans_prob * v_mf[4];

      // Hybrid value at level 1 (mix MB and MF) w=1 purely mb, 1=0 purely mf
      v_hybrid[1] = w[i] * v_mb[1] + (1 - w[i]) * v_mf[1];
      v_hybrid[2] = w[i] * v_mb[2] + (1 - w[i]) * v_mf[2];

      // Level-1 choice probability & likelihood
      level1_choice_01 = level1_choice[i,t] - 1; // 1->0, 2->1 (Bernoulli expects 0/1)
      if (t == 1) {
        level1_prob_choice2 = inv_logit( beta[i] * ( v_hybrid[2] - v_hybrid[1] ) );
      } else {
        // Perseveration term uses previous observed choice; scaling matches original model
        level1_prob_choice2 = inv_logit( beta[i] * ( v_hybrid[2] - v_hybrid[1] )
                                         + pi[i] * ( 2 * level1_choice[i,t-1] - 3 ) );
      }
      // observation model for level 1 choice
      level1_choice_01 ~ bernoulli( level1_prob_choice2 );

      // Level-2 choice probability & likelihood (CHANGED 1..2)
      level2_choice_01 = level2_choice[i,t] - 1; // 1->0, 2->1
      // Softmax/logit comparing the two second-stage action values (indices 3 and 4)
      level2_prob_choice2 = inv_logit( beta[i] * ( v_mf[4] - v_mf[3] ) );
      level2_choice_01 ~ bernoulli( level2_prob_choice2 );

      // Value updates after observing the level-2 choice and reward
      // Update level-1 MF for the chosen first-stage stimulus using chosen level2 value
      //v_mf[level1_choice[i,t]] += a[i] * ( v_mf[2 + level2_choice[i,t]] - v_mf[level1_choice[i,t]] );

      // Update the chosen level-2 MF value with the experienced reward
      v_mf[2 + level2_choice[i,t]] += a[i] * ( reward[i,t] - v_mf[2 + level2_choice[i,t]] );

      // Update level-1 MF with direct reward bootstrap (MAYBE REMOVE THIS LINE - SEEMS like a double update)
      v_mf[level1_choice[i,t]] += a[i] * ( reward[i,t] - v_mf[2 + level2_choice[i,t]] );

    } 
  } 
}

// posterior predictive, RPE regressors, log-lik
generated quantities {
  // population-level (transformed) means (those for reporting!)
  real<lower=0,upper=1> mu_a;
  real<lower=0>         mu_beta;
  real<lower=0,upper=5> mu_pi;
  real<lower=0,upper=1> mu_w;

  // Model regressors and predictive parameters
  real mf_RPE[N, T];    // model-free RPE at level 1 (per trial)
  real mb_RPE[N, T];    // model-based RPE at level 1
  real mfb_RPE[N, T];   // difference between MF and MB RPE (mf - mb)

  real log_lik[N];      // subject log-likelihood (sum across trials)

  real y_pred_step1[N, T]; // posterior predictive choices level 1 (0/1 stored as -1/1? we store 0/1)
  real y_pred_step2[N, T]; // posterior predictive choices level 2

  // initialize outputs to safe values
  for (i in 1:N) {
    log_lik[i] = 0;
    for (t in 1:T) {
      mf_RPE[i, t] = 0;
      mb_RPE[i, t] = 0;
      mfb_RPE[i, t] = 0;
      y_pred_step1[i, t] = -1;
      y_pred_step2[i, t] = -1;
    }
  }

  // Generate group level parameter values
  mu_a     = Phi_approx( mu_pr[1] );
  mu_beta  = exp( mu_pr[2] );
  mu_pi     = Phi_approx( mu_pr[3] ) * 5;
  mu_w      = Phi_approx( mu_pr[4] );

  { // local block for generating trialwise regressors and predictive draws
    for (i in 1:N) {
      vector[2] v_mb;
      vector[4] v_mf;
      vector[2] v_hybrid;
      real level1_prob_choice2;
      real level2_prob_choice2;
      int level1_choice_01;
      int level2_choice_01;

      // initialize
      v_mb     = rep_vector(0.0, 2);
      v_mf     = rep_vector(0.0, 4);
      v_hybrid = rep_vector(0.0, 2);

      for (t in 1:Tsubj[i]) {

      // compute model-based values (change to reduced indices 3 & 4)
      v_mb[1] = trans_prob * v_mf[3] + (1 - trans_prob) * v_mf[4];
      v_mb[2] = (1 - trans_prob) * v_mf[3] + trans_prob * v_mf[4];

      // hybrid values
      v_hybrid[1] = w[i] * v_mb[1] + (1 - w[i]) * v_mf[1];
      v_hybrid[2] = w[i] * v_mb[2] + (1 - w[i]) * v_mf[2];

      // level 1 choice probability
      level1_choice_01 = level1_choice[i,t] - 1;
      if (t == 1)
      level1_prob_choice2 = inv_logit( beta[i] * ( v_hybrid[2] - v_hybrid[1] ) );
      else
      level1_prob_choice2 = inv_logit( beta[i] * ( v_hybrid[2] - v_hybrid[1] ) + pi[i] * ( 2 * level1_choice[i,t-1] - 3 ) );
      log_lik[i] += bernoulli_lpmf( level1_choice_01 | level1_prob_choice2 );

      // level 2 choice probability
      level2_choice_01 = level2_choice[i,t] - 1;
      level2_prob_choice2 = inv_logit( beta[i] * ( v_mf[4] - v_mf[3] ) );
      log_lik[i] += bernoulli_lpmf( level2_choice_01 | level2_prob_choice2 );

      // posterior predictive draws
      y_pred_step1[i,t] = bernoulli_rng(level1_prob_choice2);
      y_pred_step2[i,t] = bernoulli_rng(level2_prob_choice2);

      // store RPEs (before update! )
      mf_RPE[i, t]  = reward[i, t] - v_mf[level1_choice[i, t]];
      mb_RPE[i, t]  = reward[i, t] - v_mb[level1_choice[i, t]];
      mfb_RPE[i, t] = mf_RPE[i, t] - mb_RPE[i, t];

      // MF updates
      //v_mf[level1_choice[i,t]] += a[i] * ( v_mf[2 + level2_choice[i,t]] - v_mf[level1_choice[i,t]] );
      v_mf[2 + level2_choice[i,t]] += a[i] * ( reward[i,t] - v_mf[2 + level2_choice[i,t]] );
      v_mf[level1_choice[i,t]] += a[i] * ( reward[i,t] - v_mf[2 + level2_choice[i,t]] );
              
      } 
    } 
  } 
}

