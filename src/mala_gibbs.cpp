#include <Rcpp.h>
#include "log_pl_ratio.h"
using namespace Rcpp;

// ----------------------------------------------------------------------------|
// The gradient of threshold parameters for a regular ordinal variable.
// ----------------------------------------------------------------------------|
NumericVector gradient_thresholds(int variable,
                                  NumericVector thresholds,
                                  IntegerMatrix n_cat_obs,
                                  IntegerVector no_categories,
                                  NumericMatrix rest_matrix,
                                  double threshold_alpha,
                                  double threshold_beta) {

  int no_persons = rest_matrix.nrow();
  int no_thresholds = no_categories[variable];

  NumericVector gradient (no_thresholds);
  double bound = 0.0;
  double bound_s = 0.0;
  double denominator = 0.0;
  double numerator = 0.0;
  double rest_score = 0.0;
  double exponent = 0.0;
  double tmp;
  int score = 0;

  //Contribution of the pseudolikelihood ---------------------------------------
  for(int category = 0; category < no_categories[variable]; category++) {
    gradient[category] = n_cat_obs(category + 1, variable);
  }

  bound_s = thresholds[0];
  for(int category = 1; category < no_categories[variable]; category++) {
    if(thresholds[category] > bound_s) {
      bound_s = thresholds[category];
    }
  }
  for(int person = 0; person < no_persons; person++) {
    rest_score = rest_matrix(person, variable);
    if(rest_score > 0) {
      bound = bound_s + no_categories[variable] * rest_score;
    } else {
      bound = bound_s;
    }

    denominator = std::exp(-bound);
    for(int category = 0; category < no_categories[variable]; category++) {
      score = category + 1;
      exponent = thresholds[category] +
        score * rest_score -
        bound;
      denominator += std::exp(exponent);
    }

    for(int category = 0; category < no_categories[variable]; category++) {
      score = category + 1;
      exponent = thresholds[category] +
        score * rest_score -
        bound;
      numerator = std::exp(exponent);
      gradient[category] -= numerator / denominator;
    }
  }

  //Contribution of the prior density ------------------------------------------
  for(int category = 0; category < no_categories[variable]; category++) {
    gradient[category] += threshold_alpha;
    tmp = threshold_alpha + threshold_beta;
    tmp *= std::exp(thresholds[category]);
    tmp /= (1 + std::exp(thresholds[category]));
    gradient[category] -= tmp;
  }

  return gradient;
}

// ----------------------------------------------------------------------------|
// The unnormalized log posterior density of the threshold parameters for a
//  regular ordinal variable.
// ----------------------------------------------------------------------------|
double log_posterior_thresholds(int variable,
                                NumericVector thresholds,
                                IntegerMatrix n_cat_obs,
                                IntegerVector no_categories,
                                NumericMatrix rest_matrix,
                                double threshold_alpha,
                                double threshold_beta) {
  int no_persons = rest_matrix.nrow();
  double rest_score = 0.0;
  double bound =  0.0;
  double log_posterior = 0.0;
  double denominator = 0.0;
  double exponent = 0.0;
  int score = 0;

  //Contribution of the pseudolikelihood ---------------------------------------
  for(int category = 0; category < no_categories[variable]; category++) {
    log_posterior += n_cat_obs(category + 1, variable) * thresholds[category];
  }
  for(int person = 0; person < no_persons; person++) {
    rest_score = rest_matrix(person, variable);
    if(rest_score > 0) {
      bound = no_categories[variable] * rest_score;
    } else {
      bound = 0.0;
    }

    log_posterior -= bound;
    denominator = std::exp(-bound);
    for(int category = 0; category < no_categories[variable]; category++) {
      score = category + 1;
      exponent = thresholds[category] +
        score * rest_score -
        bound;
      denominator += std::exp(exponent);
    }
    log_posterior -= log(denominator);
  }

  //Contribution of the prior densities ----------------------------------------
  for(int category = 0; category < no_categories[variable]; category++) {
    log_posterior -= R::lbeta(threshold_alpha, threshold_beta);
    log_posterior += threshold_alpha * thresholds[category];
    log_posterior -= (threshold_alpha + threshold_beta) * std::log(1 + std::exp(thresholds[category]));
  }
  return log_posterior;
}

// ----------------------------------------------------------------------------|
// Adaptive MALA to sample from the full-conditional of the threshold parameters
//   for a regular binary or ordinal variable
// ----------------------------------------------------------------------------|
void metropolis_thresholds_regular_mala(NumericMatrix thresholds,
                                        int variable,
                                        IntegerVector no_categories,
                                        IntegerMatrix n_cat_obs,
                                        double threshold_alpha,
                                        double threshold_beta,
                                        NumericMatrix rest_matrix,
                                        NumericVector threshold_step_size,
                                        double phi,
                                        double target_ar,
                                        int t,
                                        double epsilon_lo,
                                        double epsilon_hi) {
  int no_thresholds = no_categories[variable];
  double step_size = threshold_step_size[variable];
  double norm_sample;
  NumericVector proposed(no_thresholds);
  NumericVector current(no_thresholds);

  for(int category = 0; category < no_categories[variable]; category++) {
    current[category] = thresholds(variable, category);
  }

  NumericVector gradient_current = gradient_thresholds(variable,
                                                       current,
                                                       n_cat_obs,
                                                       no_categories,
                                                       rest_matrix,
                                                       threshold_alpha,
                                                       threshold_beta);

  for(int category = 0; category < no_categories[variable]; category++) {
    norm_sample = R::rnorm(0.0, 1.0);
    proposed[category] = thresholds(variable, category);
    proposed[category] += gradient_current[category] * step_size * step_size / 2;
    proposed[category] += step_size * norm_sample;
  }

  NumericVector gradient_proposed = gradient_thresholds(variable,
                                                        proposed,
                                                        n_cat_obs,
                                                        no_categories,
                                                        rest_matrix,
                                                        threshold_alpha,
                                                        threshold_beta);



  double proposed_log_posterior = log_posterior_thresholds(variable,
                                                           proposed,
                                                           n_cat_obs,
                                                           no_categories,
                                                           rest_matrix,
                                                           threshold_alpha,
                                                           threshold_beta);

  double current_log_posterior = log_posterior_thresholds(variable,
                                                          current,
                                                          n_cat_obs,
                                                          no_categories,
                                                          rest_matrix,
                                                          threshold_alpha,
                                                          threshold_beta);

  double current_to_proposed = 0.0;
  double tmp = 0.0;
  step_size = threshold_step_size[variable];
  for(int category = 0; category < no_categories[variable]; category++) {
    tmp = proposed[category];
    tmp -= current[category];
    tmp -= gradient_current[category] * step_size * step_size / 2;
    current_to_proposed += tmp * tmp;
  }
  current_to_proposed *= -.5;
  current_to_proposed /= (step_size * step_size);

  double proposed_to_current = 0.0;
  for(int category = 0; category < no_categories[variable]; category++) {
    tmp = current[category];
    tmp -= proposed[category];
    tmp -= gradient_proposed[category] * step_size * step_size / 2;
    proposed_to_current += tmp * tmp;
  }
  proposed_to_current *= -.5;
  proposed_to_current /= (step_size * step_size);

  double log_mh_ratio = proposed_log_posterior;
  log_mh_ratio -= current_log_posterior;
  log_mh_ratio += proposed_to_current;
  log_mh_ratio -= current_to_proposed;


  double U = std::log(R::unif_rand());
  if(U < log_mh_ratio) {
    for(int category = 0; category < no_categories[variable]; category++) {
      thresholds(variable, category) = proposed[category];
    }
  }

  //Robbins-Monro adaptation ---------------------------------------------------
  double prob;
  if(log_mh_ratio > 0) {
    prob = 1.0;
  } else {
    prob = std::exp(log_mh_ratio);
  }

  double new_step_size = step_size;
  new_step_size += (prob - target_ar) * std::exp(-log(t) * phi);

  if(std::isnan(new_step_size) == true) {
    new_step_size = 1.0;
  }

  if(new_step_size < epsilon_lo) {
    new_step_size = epsilon_lo;
  } else if (new_step_size > epsilon_hi) {
    new_step_size = epsilon_hi;
  }
  threshold_step_size[variable] = new_step_size;
}



// ----------------------------------------------------------------------------|
// Function to compute the gradient of the interaction parameters
// ----------------------------------------------------------------------------|
double gradient_one_interaction(double interaction,
                                NumericMatrix thresholds,
                                IntegerMatrix observations,
                                IntegerVector no_categories,
                                double interaction_scale,
                                int variable1,
                                int variable2,
                                NumericMatrix interactions,
                                NumericMatrix rest_matrix) {

  int no_persons = observations.nrow();

  double gradient = 0.0;
  double bound = 0.0;
  double bound_1 = 0.0;
  double bound_2 = 0.0;
  double denominator = 0.0;
  double numerator = 0.0;
  double rest_score = 0.0;
  double exponent = 0.0;
  int score = 0;

  bound_1 = thresholds(variable1, 0);
  for(int category = 1; category < no_categories[variable1]; category++) {
    if(thresholds(variable1, category) > bound_1) {
      bound_1 = thresholds(variable1, category);
    }
  }
  bound_2 = thresholds(variable2, 0);
  for(int category = 1; category < no_categories[variable2]; category++) {
    if(thresholds(variable2, category) > bound_2) {
      bound_2 = thresholds(variable2, category);
    }
  }

  for(int person = 0; person < no_persons; person++) {
    // Sufficient statistic
    gradient += 2 * observations(person, variable1) * observations(person, variable2);

    // Contribution of the pseudolikelihood of variable1
    rest_score = rest_matrix(person, variable1);
    rest_score -= observations(person, variable2) * interactions(variable1, variable2);
    rest_score += observations(person, variable2) * interaction;

    if(rest_score > 0) {
      bound = bound_1 + no_categories[variable1] * rest_score;
    } else {
      bound = bound_1;
    }

    denominator = std::exp(-bound);
    numerator = 0.0;
    for(int category = 0; category < no_categories[variable1]; category++) {
      score = category + 1;
      exponent = thresholds(variable1, category) + score * rest_score - bound;
      numerator += score * observations(person, variable2) * std::exp(exponent);
      denominator += std::exp(exponent);
    }
    gradient -= numerator / denominator;

    // Contribution of the pseudolikelihood of variable2
    rest_score = rest_matrix(person, variable2);
    rest_score -= observations(person, variable1) * interactions(variable1, variable2);
    rest_score += observations(person, variable1) * interaction;

    if(rest_score > 0) {
      bound = bound_2 + no_categories[variable2] * rest_score;
    } else {
      bound = bound_2;
    }

    denominator = std::exp(-bound);
    numerator = 0.0;
    for(int category = 0; category < no_categories[variable2]; category++) {
      score = category + 1;
      exponent = thresholds(variable2, category) + score * rest_score - bound;
      numerator += score * observations(person, variable1) * std::exp(exponent);
      denominator += std::exp(exponent);
    }
    gradient -= numerator / denominator;
  }

  // Contribution of the prior density
  gradient -= 2 * interaction /
    (interaction* interaction + interaction_scale * interaction_scale);

  return gradient;
}

// ----------------------------------------------------------------------------|
// Function to compute the unnormalized log posterior density of the interaction parameters
// ----------------------------------------------------------------------------|
double log_posterior_one_interaction(double interaction,
                                     NumericMatrix thresholds,
                                     IntegerMatrix observations,
                                     double interaction_scale,
                                     IntegerVector no_categories,
                                     int variable1,
                                     int variable2,
                                     NumericMatrix interactions,
                                     NumericMatrix rest_matrix) {

  int no_persons = observations.nrow();
  double rest_score = 0.0;
  double bound =  0.0;
  double log_posterior = 0.0;
  double denominator = 0.0;
  double exponent = 0.0;
  int score = 0;

  for(int person = 0; person < no_persons; person++) {
    //contributon pseudolikelihood of variable 1 -------------------------------
    rest_score = rest_matrix(person, variable1);
    rest_score -= observations(person, variable2) * interactions(variable1, variable2);
    rest_score += observations(person, variable2) * interaction;

    log_posterior += observations(person, variable1) * rest_score;
    bound = no_categories[variable1] * rest_score;
    log_posterior -= bound;
    denominator = std::exp(-bound);
    for(int category = 0; category < no_categories[variable1]; category++) {
      if(observations(person, variable1) == category + 1) {
        log_posterior += thresholds(variable1, category);
      }
      score = category + 1;
      exponent = thresholds(variable1, category) + score * rest_score - bound;
      denominator += std::exp(exponent);
    }
    log_posterior -= log(denominator);

    //contributon pseudolikelihood of variable 2 -------------------------------
    rest_score = rest_matrix(person, variable2);
    rest_score -= observations(person, variable1) * interactions(variable1, variable2);
    rest_score += observations(person, variable1) * interaction;

    log_posterior += observations(person, variable2) * rest_score;
    bound = no_categories[variable2] * rest_score;
    log_posterior -= bound;
    denominator = std::exp(-bound);
    for(int category = 0; category < no_categories[variable2]; category++) {
      if(observations(person, variable2) == category + 1) {
        log_posterior += thresholds(variable2, category);
      }
      score = category + 1;
      exponent = thresholds(variable2, category) + score * rest_score - bound;
      denominator += std::exp(exponent);
    }
    log_posterior -= log(denominator);
  }

  // Contribution of the prior density
  log_posterior += R::dcauchy(interaction, 0.0, interaction_scale, true);

  return log_posterior;
}

// ----------------------------------------------------------------------------|
// MH algorithm to sample from the full-conditional of the interaction parameters using MALA
// ----------------------------------------------------------------------------|
void mala_interactions(NumericMatrix interactions,
                       NumericMatrix thresholds,
                       IntegerMatrix observations,
                       IntegerMatrix indicator,
                       IntegerVector no_categories,
                       NumericMatrix rest_matrix,
                       double interaction_scale,
                       NumericMatrix interactions_step_size,
                       double phi,
                       double target_ar,
                       int t,
                       double epsilon_lo,
                       double epsilon_hi) {

  int no_persons = observations.nrow();
  int no_variables = observations.ncol();
  double step_size;
  double norm_sample;
  double current_state;
  double proposed_state;

  for(int variable1 = 0; variable1 < no_variables - 1; variable1++) {
    for(int variable2 = variable1 + 1; variable2 < no_variables; variable2++) {
      if(indicator(variable1, variable2) == 1) {
        step_size = interactions_step_size(variable1, variable2);
        current_state = interactions(variable1, variable2);

        double gradient_current = gradient_one_interaction(current_state,
                                                           thresholds,
                                                           observations,
                                                           no_categories,
                                                           interaction_scale,
                                                           variable1,
                                                           variable2,
                                                           interactions,
                                                           rest_matrix);

        double current_log_posterior = log_posterior_one_interaction(current_state,
                                                                     thresholds,
                                                                     observations,
                                                                     interaction_scale,
                                                                     no_categories,
                                                                     variable1,
                                                                     variable2,
                                                                     interactions,
                                                                     rest_matrix);

        norm_sample = R::rnorm(0.0, 1.0);
        proposed_state = current_state + gradient_current * step_size * step_size / 2 + step_size * norm_sample;

        double gradient_proposed = gradient_one_interaction(proposed_state,
                                                            thresholds,
                                                            observations,
                                                            no_categories,
                                                            interaction_scale,
                                                            variable1,
                                                            variable2,
                                                            interactions,
                                                            rest_matrix);

        double proposed_log_posterior = log_posterior_one_interaction(proposed_state,
                                                                      thresholds,
                                                                      observations,
                                                                      interaction_scale,
                                                                      no_categories,
                                                                      variable1,
                                                                      variable2,
                                                                      interactions,
                                                                      rest_matrix);

        double state_diff = proposed_state - current_state;
        double current_to_proposed = 0.0;
        double proposed_to_current = 0.0;
        double tmp = 0.0;

        tmp = state_diff - gradient_current * step_size * step_size / 2;
        current_to_proposed = -tmp * tmp / (2 * step_size * step_size);

        tmp = -state_diff - gradient_proposed * step_size * step_size / 2;
        proposed_to_current = -tmp * tmp / (2 * step_size * step_size);

        double log_mh_ratio = proposed_log_posterior - current_log_posterior + proposed_to_current - current_to_proposed;

        double U = std::log(R::unif_rand());
        if(U < log_mh_ratio) {
          interactions(variable1, variable2) = proposed_state;
          interactions(variable2, variable1) = proposed_state;

          // Update the matrix of rest scores
          for(int person = 0; person < no_persons; person++) {
            rest_matrix(person, variable1) += observations(person, variable2) * state_diff;
            rest_matrix(person, variable2) += observations(person, variable1) * state_diff;
          }
        }

        double prob;
        if(log_mh_ratio > 0) {
          prob = 1.0;
        } else {
          prob = std::exp(log_mh_ratio);
        }

        double new_step_size = step_size + (prob - target_ar) * std::exp(-std::log(t) * phi);

        if(std::isnan(new_step_size)) {
          new_step_size = 1.0;
        } else if(new_step_size < epsilon_lo) {
          new_step_size = epsilon_lo;
        } else if (new_step_size > epsilon_hi) {
          new_step_size = epsilon_hi;
        }
        interactions_step_size(variable1, variable2) = new_step_size;
        interactions_step_size(variable2, variable1) = new_step_size;
      }
    }
  }
}

// ----------------------------------------------------------------------------|
// MH algorithm to sample from the full-conditional of an edge + interaction
//  pair for Bayesian edge selection
// ----------------------------------------------------------------------------|
void mala_edge_interaction_pair(NumericMatrix interactions,
                                NumericMatrix thresholds,
                                IntegerMatrix gamma,
                                IntegerMatrix observations,
                                IntegerVector no_categories,
                                NumericMatrix interactions_step_size,
                                double interaction_scale,
                                IntegerMatrix index,
                                int no_interactions,
                                int no_persons,
                                NumericMatrix rest_matrix,
                                NumericMatrix theta,
                                LogicalVector variable_bool,
                                IntegerVector reference_category) {
  double proposed_state;
  double current_state;
  double log_prob;
  double U;
  double norm_sample;
  double step_size;

  int variable1;
  int variable2;

  for(int cntr = 0; cntr < no_interactions; cntr ++) {
    variable1 = index(cntr, 1) - 1;
    variable2 = index(cntr, 2) - 1;

    step_size = interactions_step_size(variable1, variable2);
    current_state = interactions(variable1, variable2);

    double gradient = gradient_one_interaction(0.0,
                                               thresholds,
                                               observations,
                                               no_categories,
                                               interaction_scale,
                                               variable1,
                                               variable2,
                                               interactions,
                                               rest_matrix);

    if(gamma(variable1, variable2) == 0) {
      norm_sample = R::rnorm(0.0, 1.0);
      proposed_state = current_state;
      proposed_state += gradient * step_size * step_size / 2;
      proposed_state += step_size * norm_sample;
    } else {
      proposed_state = 0.0;
    }

    log_prob = log_pseudolikelihood_ratio(interactions,
                                          thresholds,
                                          observations,
                                          no_categories,
                                          no_persons,
                                          variable1,
                                          variable2,
                                          proposed_state,
                                          current_state,
                                          rest_matrix,
                                          variable_bool,
                                          reference_category);


    if(gamma(variable1, variable2) == 0) {
      log_prob += R::dcauchy(proposed_state, 0.0, interaction_scale, true);
      log_prob -= R::dnorm(proposed_state,
                           current_state + gradient * step_size * step_size / 2,
                           step_size,
                           true);
      log_prob += log(theta(variable1, variable2) / (1 - theta(variable1, variable2)));
    } else {
      log_prob -= R::dcauchy(current_state, 0.0, interaction_scale, true);
      log_prob -= R::dnorm(current_state,
                           proposed_state + gradient * step_size * step_size / 2,
                           step_size,
                           true);
      log_prob -= log(theta(variable1, variable2) / (1 - theta(variable1, variable2)));
    }

    U = R::unif_rand();
    if(std::log(U) < log_prob) {
      gamma(variable1, variable2) = 1 - gamma(variable1, variable2);
      gamma(variable2, variable1) = 1 - gamma(variable2, variable1);

      interactions(variable1, variable2) = proposed_state;
      interactions(variable2, variable1) = proposed_state;

      double state_diff = proposed_state - current_state;

      //Update the matrix of rest scores ---------------------------------------
      for(int person = 0; person < no_persons; person++) {
        rest_matrix(person, variable1) += observations(person, variable2) *
          state_diff;
        rest_matrix(person, variable2) += observations(person, variable1) *
          state_diff;
      }
    }
  }
}

