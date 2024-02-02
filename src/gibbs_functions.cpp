// [[Rcpp::depends(RcppProgress)]]
#include <Rcpp.h>
#include <progress.hpp>
#include <progress_bar.hpp>
using namespace Rcpp;

// ----------------------------------------------------------------------------|
// Impute missing data from full-conditional
// ----------------------------------------------------------------------------|
List impute_missing_data(NumericMatrix interactions,
                         NumericMatrix thresholds,
                         IntegerMatrix observations,
                         IntegerMatrix n_cat_obs,
                         IntegerVector no_categories,
                         NumericMatrix rest_matrix,
                         IntegerMatrix missing_index,
                         StringVector variable_type,
                         IntegerVector ordinal_BC_RefCat) {

  int no_nodes = observations.ncol();
  int no_missings = missing_index.nrow();
  int max_no_categories = 0;
  for(int node = 0; node < no_nodes; node++) {
    if(no_categories[node] > max_no_categories) {
      max_no_categories = no_categories[node];
    }
  }
  NumericVector probabilities(max_no_categories + 1);
  double exponent, rest_score, cumsum, u;
  int score, person, node, new_observation, old_observation;

  for(int missing = 0; missing < no_missings; missing++) {
    //Which observation to impute? ---------------------------------------------
    person = missing_index(missing, 0) - 1; //R to C++ indexing
    node = missing_index(missing, 1) - 1; //R to C++ indexing

    //Generate new observation -------------------------------------------------
    rest_score = rest_matrix(person, node);

    //Two distinct (ordinal) variable types ------------------------------------
    if(variable_type[node] == "ordinal") {

      //Regular binary or ordinal MRF variable ---------------------------------
      cumsum = 1.0;
      probabilities[0] = 1.0;
      for(int category = 0; category < no_categories[node]; category++) {
        exponent = thresholds(node, category);
        exponent += (category + 1) * rest_score;
        cumsum += std::exp(exponent);
        probabilities[category + 1] = cumsum;
      }

    } else if (variable_type[node] == "ordinal_BC"){

      //Blume-Capel ordinal MRF variable ---------------------------------------
      exponent = thresholds(node, 1) *
        ordinal_BC_RefCat[node] *
        ordinal_BC_RefCat[node];
      cumsum = std::exp(exponent);
      probabilities[0] = cumsum;
      for(int category = 0; category < no_categories[node]; category++) {
        exponent = thresholds(node, 0) * (category + 1);
        exponent += thresholds(node, 1) *
          (category + 1 - ordinal_BC_RefCat[node]) *
          (category + 1 - ordinal_BC_RefCat[node]);
        exponent += (category + 1) * rest_score;
        cumsum += std::exp(exponent);
        probabilities[category + 1] = cumsum;
      }
    }

    u = cumsum * R::unif_rand();
    score = 0;
    while (u > probabilities[score]) {
      score++;
    }

    //Update observations
    new_observation = score;
    old_observation = observations(person, node);
    if(old_observation != new_observation) {
      observations(person, node) = new_observation;
      n_cat_obs(old_observation, node)--;
      n_cat_obs(new_observation, node)++;
      for(int vertex = 0; vertex < no_nodes; vertex++) {
        //interactions(i, i) = 0
        rest_matrix(person, vertex) -= old_observation * interactions(vertex, node);
        rest_matrix(person, vertex) += new_observation * interactions(vertex, node);
      }
    }
  }

  return List::create(Named("observations") = observations,
                        Named("n_cat_obs") = n_cat_obs,
                        Named("rest_matrix") = rest_matrix);
}

// ----------------------------------------------------------------------------|
// MH algorithm to sample from the full-conditional of the threshold parameters
//   for a regular binary or ordinal variable
// ----------------------------------------------------------------------------|
void metropolis_thresholds_regular(NumericMatrix interactions,
                                   NumericMatrix thresholds,
                                   IntegerMatrix observations,
                                   IntegerVector no_categories,
                                   IntegerMatrix n_cat_obs,
                                   int no_persons,
                                   int node,
                                   double threshold_alpha,
                                   double threshold_beta,
                                   NumericMatrix rest_matrix) {

  NumericVector g(no_persons);
  NumericVector q(no_persons);

  double log_prob, rest_score;
  double a, b, c;
  double tmp;
  double current_state, proposed_state;
  double U;
  double exp_current, exp_proposed;

  for(int category = 0; category < no_categories[node]; category++) {
    current_state = thresholds(node, category);
    exp_current = std::exp(current_state);
    c = (threshold_alpha + threshold_beta) / (1 + exp_current);
    for(int person = 0; person < no_persons; person++) {
      g[person] = 1.0;
      q[person] = 1.0;
      rest_score = rest_matrix(person, node);
      for(int cat = 0; cat < no_categories[node]; cat++) {
        if(cat != category) {
          g[person] += std::exp(thresholds(node, cat) +
            (cat + 1) * rest_score);
        }
      }
      q[person] = std::exp((category + 1) * rest_score);
      c +=  q[person] / (g[person] + q[person] * exp_current);
    }
    c = c / ((no_persons + threshold_alpha + threshold_beta) -
      exp_current * c);

    //Proposal is generalized beta-prime.
    a = n_cat_obs(category + 1, node) + threshold_alpha;
    b = no_persons + threshold_beta - n_cat_obs(category + 1, node);
    tmp = R::rbeta(a, b);
    proposed_state = std::log(tmp / (1  - tmp) / c);
    exp_proposed = exp(proposed_state);

    //Compute log_acceptance probability for Metropolis.
    //First, we use g and q above to compute the ratio of pseudolikelihoods
    log_prob = 0;
    for(int person = 0; person < no_persons; person++) {
      log_prob += std::log(g[person] + q[person] * exp_current);
      log_prob -= std::log(g[person] + q[person] * exp_proposed);
    }
    //Second, we add the ratio of prior probabilities
    log_prob -= (threshold_alpha + threshold_beta) *
      std::log(1 + exp_proposed);
    log_prob += (threshold_alpha + threshold_beta) *
      std::log(1 + exp_current);
    //Third, we add the ratio of proposals
    log_prob -= (a + b) * std::log(1 + c * exp_current);
    log_prob += (a + b) * std::log(1 + c * exp_proposed);

    U = std::log(R::unif_rand());
    if(U < log_prob) {
      thresholds(node, category) = proposed_state;
    }
  }
}

// ----------------------------------------------------------------------------|
// MH algorithm to sample from the full-conditional of the threshold parameters
//   for a Blume-Capel ordinal variable
// ----------------------------------------------------------------------------|
void metropolis_thresholds_blumecapel(NumericMatrix interactions,
                                      NumericMatrix thresholds,
                                      IntegerMatrix observations,
                                      IntegerVector no_categories,
                                      int no_persons,
                                      int node,
                                      IntegerVector ordinal_BC_RefCat,
                                      double threshold_alpha,
                                      double threshold_beta,
                                      NumericMatrix rest_matrix) {

  NumericVector g(no_categories[node] + 1);
  double log_prob;
  double a, b, d;
  double tmp;
  double current_state, proposed_state;
  double U;
  double exp_current, exp_proposed;
  double numerator, denominator;
  int t = 0;

  //Update Blume-Capel alpha parameter -----------------------------------------
  current_state = thresholds(node, 0);
  exp_current = std::exp(current_state);

  for(int category = 0; category == no_categories[node]; category++) {
    g[category] = std::exp(thresholds(node, 1) *
      (category - ordinal_BC_RefCat[node]) *
      (category - ordinal_BC_RefCat[node]));
  }

  d = (threshold_alpha + threshold_beta) / (1 + exp_current);
  for(int person = 0; person < no_persons; person++) {
    t += observations(person, node);
    numerator = 0.0;
    denominator = g[0];
    for(int category = 0; category < no_categories[node]; category++) {
      tmp = rest_matrix(person, node) *
        (category + 1);
      numerator += (category + 1) * std::exp(tmp + current_state *
        category) * g[category + 1];
      denominator += std::exp(tmp + current_state *
        (category + 1)) * g[category + 1];
    }
    d += numerator / denominator;
  }
  d /= (threshold_alpha +
    threshold_beta +
    no_persons * no_categories[node] -
    exp_current * d);

  //Proposal is generalized beta-prime.
  a = t + threshold_alpha;
  b = no_persons * no_categories[node] + threshold_beta - t;
  tmp = R::rbeta(a, b);
  proposed_state = std::log(tmp / (1  - tmp) / d);
  exp_proposed = exp(proposed_state);

  //Compute log_acceptance probability for Metropolis.
  log_prob = 0;
  for(int person = 0; person < no_persons; person++) {
    numerator = g[0];
    denominator = g[0];
    for(int category = 0; category < no_categories[node]; category++) {
      tmp = rest_matrix(person, node) *
        (category + 1);
      numerator += std::exp(tmp + current_state *
        (category + 1)) * g[category + 1];
      denominator += std::exp(tmp + proposed_state *
        (category + 1)) * g[category + 1];
    }
    log_prob += std::log(numerator);
    log_prob -= std::log(denominator);
  }
  //Second, we add the ratio of prior probabilities
  log_prob -= (threshold_alpha + threshold_beta) *
    std::log(1 + exp_proposed);
  log_prob += (threshold_alpha + threshold_beta) *
    std::log(1 + exp_current);
  //Third, we add the ratio of proposals
  log_prob -= (a + b) * std::log(1 + d * exp_current);
  log_prob += (a + b) * std::log(1 + d * exp_proposed);

  U = std::log(R::unif_rand());
  if(U < log_prob) {
    thresholds(node, 0) = proposed_state;
  }

  //Update Blume-Capel beta parameter ------------------------------------------
  current_state = thresholds(node, 1);
  exp_current = std::exp(current_state);

  for(int category = 0; category == no_categories[node]; category++) {
    g[category] = (category - ordinal_BC_RefCat[node]) *
      (category - ordinal_BC_RefCat[node]);
  }

  t = 0;
  d = (threshold_alpha + threshold_beta) / (1 + exp_current);
  for(int person = 0; person < no_persons; person++) {
    t += (observations(person, node) - ordinal_BC_RefCat[node]) *
      (observations(person, node) - ordinal_BC_RefCat[node]);
    numerator = g[0] * std::exp(current_state * (g[0] - 1));
    denominator = std::exp(current_state * g[0]);
    for(int category = 0; category < no_categories[node]; category++) {
      tmp = (thresholds(node, 0) + rest_matrix(person, node)) *
        (category + 1);
      numerator += g[category + 1] * std::exp(tmp + current_state *
      (g[category + 1] - 1));
      denominator += std::exp(tmp + current_state *
        g[category + 1]);
    }
    d += numerator / denominator;
  }
  d /= (threshold_alpha +
    threshold_beta +
    no_persons * max(g) -
    exp_current * d);

  //Proposal is generalized beta-prime.
  a = t + threshold_alpha;
  b = no_persons * max(g) + threshold_beta - t;
  tmp = R::rbeta(a, b);
  proposed_state = std::log(tmp / (1  - tmp) / d);
  exp_proposed = exp(proposed_state);

  //Compute log_acceptance probability for Metropolis.
  log_prob = 0;
  for(int person = 0; person < no_persons; person++) {
    numerator = std::exp(current_state * g[0]);
    denominator = std::exp(proposed_state * g[0]);
    for(int category = 0; category < no_categories[node]; category++) {
      tmp = (thresholds(node, 0) + rest_matrix(person, node)) * (category + 1);
      numerator += std::exp(tmp + current_state * g[category + 1]);
      denominator += std::exp(tmp + proposed_state * g[category + 1]);
    }
    log_prob += std::log(numerator);
    log_prob -= std::log(denominator);
  }

  //Second, we add the ratio of prior probabilities
  log_prob -= (threshold_alpha + threshold_beta) *
    std::log(1 + exp_proposed);
  log_prob += (threshold_alpha + threshold_beta) *
    std::log(1 + exp_current);
  //Third, we add the ratio of proposals
  log_prob -= (a + b) * std::log(1 + d * exp_current);
  log_prob += (a + b) * std::log(1 + d * exp_proposed);

  U = std::log(R::unif_rand());
  if(U < log_prob) {
    thresholds(node, 1) = proposed_state;
  }
}

// ----------------------------------------------------------------------------|
// The log pseudolikelihood ratio [proposed against current] for an interaction
// ----------------------------------------------------------------------------|
double log_pseudolikelihood_ratio(NumericMatrix interactions,
                                  NumericMatrix thresholds,
                                  IntegerMatrix observations,
                                  IntegerVector no_categories,
                                  int no_persons,
                                  int node1,
                                  int node2,
                                  double proposed_state,
                                  double current_state,
                                  NumericMatrix rest_matrix,
                                  StringVector variable_type,
                                  IntegerVector ordinal_BC_RefCat) {
  double rest_score, bound;
  double pseudolikelihood_ratio = 0.0;
  double denominator_prop, denominator_curr, exponent;
  int score, obs_score1, obs_score2;

  double delta_state = proposed_state - current_state;

  for(int person = 0; person < no_persons; person++) {
    obs_score1 = observations(person, node1);
    obs_score2 = observations(person, node2);

    pseudolikelihood_ratio += 2 * obs_score1 * obs_score2 * delta_state;

    //Node 1 log pseudolikelihood ratio
    rest_score = rest_matrix(person, node1) -
      obs_score2 * interactions(node2, node1);

    if(rest_score > 0) {
      bound = no_categories[node1] * rest_score;
    } else {
      bound = 0.0;
    }

    //Two distinct (ordinal) variable types ------------------------------------
    if(variable_type[node1] == "ordinal") {
      //Regular binary or ordinal MRF variable ---------------------------------
      denominator_prop = std::exp(-bound);
      denominator_curr = std::exp(-bound);
      for(int category = 0; category < no_categories[node1]; category++) {
        score = category + 1;
        exponent = thresholds(node1, category) +
          score * rest_score -
          bound;
        denominator_prop +=
          std::exp(exponent + score * obs_score2 * proposed_state);
        denominator_curr +=
          std::exp(exponent + score * obs_score2 * current_state);
      }
    } else if (variable_type[node1] == "ordinal_BC"){
      exponent = thresholds(node1, 1) *
        (ordinal_BC_RefCat[node1]) *
        (ordinal_BC_RefCat[node1]);
      denominator_prop = std::exp(exponent - bound);
      denominator_curr = std::exp(exponent - bound);
      //Blume-Capel ordinal MRF variable ---------------------------------------
      for(int category = 0; category < no_categories[node1]; category++) {
        score = category + 1;
        exponent = thresholds(node1, 0) * score;
        exponent += thresholds(node1, 1) *
          (score - ordinal_BC_RefCat[node1]) *
          (score - ordinal_BC_RefCat[node1]);
        exponent+=  score * rest_score - bound;
        denominator_prop +=
          std::exp(exponent + score * obs_score2 * proposed_state);
        denominator_curr +=
          std::exp(exponent + score * obs_score2 * current_state);
      }
    }
    pseudolikelihood_ratio -= std::log(denominator_prop);
    pseudolikelihood_ratio += std::log(denominator_curr);

    //Node 2 log pseudolikelihood ratio
    rest_score = rest_matrix(person, node2) -
      obs_score1 * interactions(node1, node2);

    if(rest_score > 0) {
      bound = no_categories[node2] * rest_score;
    } else {
      bound = 0.0;
    }

    //Two distinct (ordinal) variable types ------------------------------------
    if(variable_type[node2] == "ordinal") {
      //Regular binary or ordinal MRF variable ---------------------------------
      denominator_prop = std::exp(-bound);
      denominator_curr = std::exp(-bound);
      for(int category = 0; category < no_categories[node2]; category++) {
        score = category + 1;
        exponent = thresholds(node2, category) +
          score * rest_score -
          bound;
        denominator_prop +=
          std::exp(exponent + score * obs_score1 * proposed_state);
        denominator_curr +=
          std::exp(exponent + score * obs_score1 * current_state);
      }
    } else if (variable_type[node2] == "ordinal_BC"){
      exponent = thresholds(node2, 1) *
        (ordinal_BC_RefCat[node2]) *
        (ordinal_BC_RefCat[node2]);
      denominator_prop = std::exp(exponent - bound);
      denominator_curr = std::exp(exponent - bound);
      //Blume-Capel ordinal MRF variable ---------------------------------------
      for(int category = 0; category < no_categories[node2]; category++) {
        score = category + 1;
        exponent = thresholds(node2, 0) * score;
        exponent += thresholds(node2, 1) *
          (score - ordinal_BC_RefCat[node2]) *
          (score - ordinal_BC_RefCat[node2]);
        exponent+=  score * rest_score - bound;
        denominator_prop +=
          std::exp(exponent + score * obs_score1 * proposed_state);
        denominator_curr +=
          std::exp(exponent + score * obs_score1 * current_state);
      }
    }
    pseudolikelihood_ratio -= std::log(denominator_prop);
    pseudolikelihood_ratio += std::log(denominator_curr);
  }
  return pseudolikelihood_ratio;
}

// ----------------------------------------------------------------------------|
// MH algorithm to sample from the full-conditional of the active interaction
//  parameters for Bayesian edge selection
// ----------------------------------------------------------------------------|
void metropolis_interactions(NumericMatrix interactions,
                             NumericMatrix thresholds,
                             IntegerMatrix gamma,
                             IntegerMatrix observations,
                             IntegerVector no_categories,
                             NumericMatrix proposal_sd,
                             double cauchy_scale,
                             NumericMatrix unit_info,
                             int no_persons,
                             int no_nodes,
                             NumericMatrix rest_matrix,
                             bool adaptive,
                             double phi,
                             double target_ar,
                             int t,
                             double epsilon_lo,
                             double epsilon_hi,
                             StringVector variable_type,
                             IntegerVector ordinal_BC_RefCat,
                             String interaction_prior) {
  double proposed_state;
  double current_state;
  double log_prob;
  double U;

  for(int node1 = 0; node1 <  no_nodes - 1; node1++) {
    for(int node2 = node1 + 1; node2 <  no_nodes; node2++) {
      if(gamma(node1, node2) == 1) {
        current_state = interactions(node1, node2);
        proposed_state = R::rnorm(current_state, proposal_sd(node1, node2));

        log_prob = log_pseudolikelihood_ratio(interactions,
                                              thresholds,
                                              observations,
                                              no_categories,
                                              no_persons,
                                              node1,
                                              node2,
                                              proposed_state,
                                              current_state,
                                              rest_matrix,
                                              variable_type,
                                              ordinal_BC_RefCat);
        if(interaction_prior == "Cauchy") {
          log_prob += R::dcauchy(proposed_state, 0.0, cauchy_scale, true);
          log_prob -= R::dcauchy(current_state, 0.0, cauchy_scale, true);
        }
        if(interaction_prior == "UnitInfo") {
          log_prob += R::dnorm(proposed_state,
                               0.0,
                               unit_info(node1, node2),
                               true);
          log_prob -= R::dnorm(current_state,
                               0.0,
                               unit_info(node1, node2),
                               true);
        }


        U = R::unif_rand();
        if(std::log(U) < log_prob) {
          double state_diff = proposed_state - current_state;
          interactions(node1, node2) = proposed_state;
          interactions(node2, node1) = proposed_state;

          //Update the matrix of rest scores
          for(int person = 0; person < no_persons; person++) {
            rest_matrix(person, node1) += observations(person, node2) *
              state_diff;
            rest_matrix(person, node2) += observations(person, node1) *
              state_diff;
          }
        }

        if(adaptive == true) {
          if(log_prob > 0) {
            log_prob = 1;
          } else {
            log_prob = std::exp(log_prob);
          }
          proposal_sd(node1, node2) = proposal_sd(node1, node2) +
            (log_prob - target_ar) * std::exp(-log(t) * phi);
          if(proposal_sd(node1, node2) < epsilon_lo) {
            proposal_sd(node1, node2) = epsilon_lo;
          } else if (proposal_sd(node1, node2) > epsilon_hi) {
            proposal_sd(node1, node2) = epsilon_hi;
          }
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------|
// MH algorithm to sample from the full-conditional of an edge + interaction
//  pair for Bayesian edge selection
// ----------------------------------------------------------------------------|
void metropolis_edge_interaction_pair(NumericMatrix interactions,
                                      NumericMatrix thresholds,
                                      IntegerMatrix gamma,
                                      IntegerMatrix observations,
                                      IntegerVector no_categories,
                                      NumericMatrix proposal_sd,
                                      double cauchy_scale,
                                      NumericMatrix unit_info,
                                      IntegerMatrix index,
                                      int no_interactions,
                                      int no_persons,
                                      NumericMatrix rest_matrix,
                                      NumericMatrix theta,
                                      bool adaptive,
                                      StringVector variable_type,
                                      IntegerVector ordinal_BC_RefCat,
                                      String interaction_prior) {
  double proposed_state;
  double current_state;
  double log_prob;
  double U;

  int node1;
  int node2;

  for(int cntr = 0; cntr < no_interactions; cntr ++) {
    node1 = index(cntr, 1) - 1;
    node2 = index(cntr, 2) - 1;

    current_state = interactions(node1, node2);

    if(gamma(node1, node2) == 0) {
      proposed_state = R::rnorm(current_state, proposal_sd(node1, node2));
    } else {
      proposed_state = 0.0;
    }

    log_prob = log_pseudolikelihood_ratio(interactions,
                                          thresholds,
                                          observations,
                                          no_categories,
                                          no_persons,
                                          node1,
                                          node2,
                                          proposed_state,
                                          current_state,
                                          rest_matrix,
                                          variable_type,
                                          ordinal_BC_RefCat);

    if(gamma(node1, node2) == 0) {
      if(interaction_prior == "Cauchy") {
        log_prob += R::dcauchy(proposed_state, 0.0, cauchy_scale, true);
      }
      if(interaction_prior == "UnitInfo") {
        log_prob += R::dnorm(proposed_state,
                             0.0,
                             unit_info(node1, node2),
                             true);
      }
      log_prob -= R::dnorm(proposed_state,
                           current_state,
                           proposal_sd(node1, node2),
                           true);

      log_prob += log(theta(node1, node2) / (1 - theta(node1, node2)));
    } else {
      if(interaction_prior == "Cauchy") {
        log_prob -= R::dcauchy(current_state, 0.0, cauchy_scale, true);
      }
      if(interaction_prior == "UnitInfo") {
        log_prob -= R::dnorm(current_state,
                             0.0,
                             unit_info(node1, node2),
                             true);
      }
      log_prob += R::dnorm(current_state,
                           proposed_state,
                           proposal_sd(node1, node2),
                           true);

      log_prob -= log(theta(node1, node2) / (1 - theta(node1, node2)));
    }

    U = R::unif_rand();
    if(std::log(U) < log_prob) {
      gamma(node1, node2) = 1 - gamma(node1, node2);
      gamma(node2, node1) = 1 - gamma(node2, node1);

      interactions(node1, node2) = proposed_state;
      interactions(node2, node1) = proposed_state;

      double state_diff = proposed_state - current_state;
      //Update the matrix of rest scores
      for(int person = 0; person < no_persons; person++) {
        rest_matrix(person, node1) += observations(person, node2) *
          state_diff;
        rest_matrix(person, node2) += observations(person, node1) *
          state_diff;
      }
    }
  }
}

// ----------------------------------------------------------------------------|
// A Gibbs step for graphical model parameters for Bayesian edge selection
// ----------------------------------------------------------------------------|
List gibbs_step_gm(IntegerMatrix observations,
                   IntegerVector no_categories,
                   String interaction_prior,
                   double cauchy_scale,
                   NumericMatrix unit_info,
                   NumericMatrix proposal_sd,
                   IntegerMatrix index,
                   IntegerMatrix n_cat_obs,
                   double threshold_alpha,
                   double threshold_beta,
                   int no_persons,
                   int no_nodes,
                   int no_interactions,
                   int no_thresholds,
                   int max_no_categories,
                   IntegerMatrix gamma,
                   NumericMatrix interactions,
                   NumericMatrix thresholds,
                   NumericMatrix rest_matrix,
                   NumericMatrix theta,
                   bool adaptive,
                   double phi,
                   double target_ar,
                   int t,
                   double epsilon_lo,
                   double epsilon_hi,
                   StringVector variable_type,
                   IntegerVector ordinal_BC_RefCat,
                   bool edge_selection) {

  if(edge_selection == true) {
    //Between model move (update edge indicators and interaction parameters)
    metropolis_edge_interaction_pair(interactions,
                                     thresholds,
                                     gamma,
                                     observations,
                                     no_categories,
                                     proposal_sd,
                                     cauchy_scale,
                                     unit_info,
                                     index,
                                     no_interactions,
                                     no_persons,
                                     rest_matrix,
                                     theta,
                                     adaptive,
                                     variable_type,
                                     ordinal_BC_RefCat,
                                     interaction_prior);
  }

  //Within model move (update interaction parameters)
  metropolis_interactions(interactions,
                          thresholds,
                          gamma,
                          observations,
                          no_categories,
                          proposal_sd,
                          cauchy_scale,
                          unit_info,
                          no_persons,
                          no_nodes,
                          rest_matrix,
                          adaptive,
                          phi,
                          target_ar,
                          t,
                          epsilon_lo,
                          epsilon_hi,
                          variable_type,
                          ordinal_BC_RefCat,
                          interaction_prior);

  //Update threshold parameters
  for(int node = 0; node < no_nodes; node++) {
    if(variable_type[node] == "ordinal") {
      metropolis_thresholds_regular(interactions,
                                    thresholds,
                                    observations,
                                    no_categories,
                                    n_cat_obs,
                                    no_persons,
                                    node,
                                    threshold_alpha,
                                    threshold_beta,
                                    rest_matrix);
    }
    if (variable_type[node] == "ordinal_BC") {
      metropolis_thresholds_blumecapel(interactions,
                                       thresholds,
                                       observations,
                                       no_categories,
                                       no_persons,
                                       node,
                                       ordinal_BC_RefCat,
                                       threshold_alpha,
                                       threshold_beta,
                                       rest_matrix);
    }
  }

  return List::create(Named("gamma") = gamma,
                      Named("interactions") = interactions,
                      Named("thresholds") = thresholds,
                      Named("rest_matrix") = rest_matrix,
                      Named("proposal_sd") = proposal_sd);
}


// ----------------------------------------------------------------------------|
// The Gibbs sampler for Bayesian edge selection
// ----------------------------------------------------------------------------|
// [[Rcpp::export]]
List gibbs_sampler(IntegerMatrix observations,
                   IntegerMatrix gamma,
                   NumericMatrix interactions,
                   NumericMatrix thresholds,
                   IntegerVector no_categories,
                   String interaction_prior,
                   double cauchy_scale,
                   NumericMatrix unit_info,
                   NumericMatrix proposal_sd,
                   String edge_prior,
                   NumericMatrix theta,
                   double beta_bernoulli_alpha,
                   double beta_bernoulli_beta,
                   IntegerMatrix Index,
                   int iter,
                   int burnin,
                   IntegerMatrix n_cat_obs,
                   double threshold_alpha,
                   double threshold_beta,
                   bool na_impute,
                   IntegerMatrix missing_index,
                   StringVector variable_type,
                   IntegerVector ordinal_BC_RefCat,
                   bool adaptive = false,
                   bool save = false,
                   bool display_progress = false,
                   bool edge_selection = true) {
  int cntr;
  int no_nodes = observations.ncol();
  int no_persons = observations.nrow();
  int no_interactions = Index.nrow();
  int no_thresholds = sum(no_categories);
  int max_no_categories = max(no_categories);

  IntegerVector v = seq(0, no_interactions - 1);
  IntegerVector order(no_interactions);
  IntegerMatrix index(no_interactions, 3);

  //Parameters of adaptive proposals -------------------------------------------
  double phi = .75;
  double target_ar = 0.234;
  double epsilon_lo = 1 / no_persons;
  double epsilon_hi = 2.0;

  //The resizing based on ``save'' could probably be prettier ------------------
  int nrow = no_nodes;
  int ncol_edges = no_nodes;
  int ncol_thresholds = max_no_categories;

  if(save == true) {
    nrow = iter;
    ncol_edges= no_interactions;
    ncol_thresholds = no_thresholds;
  }

  NumericMatrix out_interactions(nrow, ncol_edges);
  NumericMatrix out_thresholds(nrow, ncol_thresholds);

  if(edge_selection == false) {
    for(int node1 = 0; node1 < no_nodes - 1; node1++) {
      for(int node2 = node1 + 1; node2 < no_nodes; node2++) {
        gamma(node1, node2) = 1;
        gamma(node2, node1) = 1;
      }
    }
    nrow = 1;
    ncol_edges = 1;
  }
  NumericMatrix out_gamma(nrow, ncol_edges);


  NumericMatrix rest_matrix(no_persons, no_nodes);
  for(int node1 = 0; node1 < no_nodes; node1++) {
    for(int person = 0; person < no_persons; person++) {
      for(int node2 = 0; node2 < no_nodes; node2++) {
        rest_matrix(person, node1) +=
          observations(person, node2) * interactions(node2, node1);
      }
    }
  }

  //Progress bar
  Progress p(iter + burnin, display_progress);

  //The Gibbs sampler ----------------------------------------------------------
  //First, we do burn-in iterations---------------------------------------------
  for(int iteration = 0; iteration < burnin; iteration++) {
    if (Progress::check_abort()) {
      return List::create(Named("gamma") = out_gamma,
                          Named("interactions") = out_interactions,
                          Named("thresholds") = out_thresholds);
    }
    p.increment();

    //Update interactions and model (between model move) -----------------------
    //Use a random order to update the edge - interaction pairs ----------------
    order = sample(v,
                   no_interactions,
                   false,
                   R_NilValue);

    for(int cntr = 0; cntr < no_interactions; cntr++) {
      index(cntr, 0) = Index(order[cntr], 0);
      index(cntr, 1) = Index(order[cntr], 1);
      index(cntr, 2) = Index(order[cntr], 2);
    }

    if(na_impute == true) {
      List out = impute_missing_data(interactions,
                                     thresholds,
                                     observations,
                                     n_cat_obs,
                                     no_categories,
                                     rest_matrix,
                                     missing_index,
                                     variable_type,
                                     ordinal_BC_RefCat);

      IntegerMatrix observations = out["observations"];
      IntegerMatrix n_cat_obs = out["n_cat_obs"];
      NumericMatrix rest_matrix = out["rest_matrix"];
    }

    List out = gibbs_step_gm(observations,
                             no_categories,
                             interaction_prior,
                             cauchy_scale,
                             unit_info,
                             proposal_sd,
                             index,
                             n_cat_obs,
                             threshold_alpha,
                             threshold_beta,
                             no_persons,
                             no_nodes,
                             no_interactions,
                             no_thresholds,
                             max_no_categories,
                             gamma,
                             interactions,
                             thresholds,
                             rest_matrix,
                             theta,
                             adaptive,
                             phi,
                             target_ar,
                             iteration + 1,
                             epsilon_lo,
                             epsilon_hi,
                             variable_type,
                             ordinal_BC_RefCat,
                             edge_selection);

    IntegerMatrix gamma = out["gamma"];
    NumericMatrix interactions = out["interactions"];
    NumericMatrix thresholds = out["thresholds"];
    NumericMatrix rest_matrix = out["rest_matrix"];
    NumericMatrix proposal_sd = out["proposal_sd"];

    if(edge_prior == "Beta-Bernoulli") {
      int sumG = 0;
      for(int i = 0; i < no_nodes - 1; i++) {
        for(int j = i + 1; j < no_nodes; j++) {
          sumG += gamma(i, j);
        }
      }
      double probability = R::rbeta(beta_bernoulli_alpha + sumG,
                                    beta_bernoulli_beta + no_interactions - sumG);

      for(int i = 0; i < no_nodes - 1; i++) {
        for(int j = i + 1; j < no_nodes; j++) {
          theta(i, j) = probability;
          theta(j, i) = probability;
        }
      }
    }

  }

  //The post burn-in iterations ------------------------------------------------
  for(int iteration = 0; iteration < iter; iteration++) {
    if (Progress::check_abort()) {
      if(edge_selection == true) {
        return List::create(Named("gamma") = out_gamma,
                            Named("interactions") = out_interactions,
                            Named("thresholds") = out_thresholds);
      } else {
        return List::create(Named("interactions") = out_interactions,
                            Named("thresholds") = out_thresholds);
      }
    }
    p.increment();

    //Update interactions and model (between model move) -----------------------
    //Use a random order to update the edge - interaction pairs ----------------
    order = sample(v,
                   no_interactions,
                   false,
                   R_NilValue);

    for(int cntr = 0; cntr < no_interactions; cntr++) {
      index(cntr, 0) = Index(order[cntr], 0);
      index(cntr, 1) = Index(order[cntr], 1);
      index(cntr, 2) = Index(order[cntr], 2);
    }

    if(na_impute == true) {
      List out = impute_missing_data(interactions,
                                     thresholds,
                                     observations,
                                     n_cat_obs,
                                     no_categories,
                                     rest_matrix,
                                     missing_index,
                                     variable_type,
                                     ordinal_BC_RefCat);

      IntegerMatrix observations = out["observations"];
      IntegerMatrix n_cat_obs = out["n_cat_obs"];
      NumericMatrix rest_matrix = out["rest_matrix"];
    }

    List out = gibbs_step_gm(observations,
                             no_categories,
                             interaction_prior,
                             cauchy_scale,
                             unit_info,
                             proposal_sd,
                             index,
                             n_cat_obs,
                             threshold_alpha,
                             threshold_beta,
                             no_persons,
                             no_nodes,
                             no_interactions,
                             no_thresholds,
                             max_no_categories,
                             gamma,
                             interactions,
                             thresholds,
                             rest_matrix,
                             theta,
                             adaptive,
                             phi,
                             target_ar,
                             iteration + 1,
                             epsilon_lo,
                             epsilon_hi,
                             variable_type,
                             ordinal_BC_RefCat,
                             edge_selection);

    IntegerMatrix gamma = out["gamma"];
    NumericMatrix interactions = out["interactions"];
    NumericMatrix thresholds = out["thresholds"];
    NumericMatrix rest_matrix = out["rest_matrix"];
    NumericMatrix proposal_sd = out["proposal_sd"];

    if(edge_prior == "Beta-Bernoulli") {
      int sumG = 0;
      for(int i = 0; i < no_nodes - 1; i++) {
        for(int j = i + 1; j < no_nodes; j++) {
          sumG += gamma(i, j);
        }
      }
      double probability = R::rbeta(beta_bernoulli_alpha + sumG,
                                    beta_bernoulli_beta + no_interactions - sumG);

      for(int i = 0; i < no_nodes - 1; i++) {
        for(int j = i + 1; j < no_nodes; j++) {
          theta(i, j) = probability;
          theta(j, i) = probability;
        }
      }
    }

    //Output -------------------------------------------------------------------
    if(save == TRUE) {
      //Save raw samples -------------------------------------------------------
      cntr = 0;
      for(int node1 = 0; node1 < no_nodes - 1; node1++) {
        for(int node2 = node1 + 1; node2 < no_nodes;node2++) {
          if(edge_selection == true) {
            out_gamma(iteration, cntr) = gamma(node1, node2);
          }
          out_interactions(iteration, cntr) = interactions(node1, node2);
          cntr++;
        }
      }
      cntr = 0;
      for(int node = 0; node < no_nodes; node++) {
        if(variable_type[node] == "ordinal") {
          for(int category = 0; category < no_categories[node]; category++) {
            out_thresholds(iteration, cntr) = thresholds(node, category);
            cntr++;
          }
        }
        if(variable_type[node] == "ordinal_BC") {
          out_thresholds(iteration, cntr) = thresholds(node, 0);
          cntr++;
          out_thresholds(iteration, cntr) = thresholds(node, 1);
          cntr++;
        }
      }
    } else {
      //Compute running averages -----------------------------------------------
      for(int node1 = 0; node1 < no_nodes - 1; node1++) {
        for(int node2 = node1 + 1; node2 < no_nodes; node2++) {
          if(edge_selection == true) {
            out_gamma(node1, node2) *= iteration;
            out_gamma(node1, node2) += gamma(node1, node2);
            out_gamma(node1, node2) /= iteration + 1;
            out_gamma(node2, node1) = out_gamma(node1, node2);
          }

          out_interactions(node1, node2) *= iteration;
          out_interactions(node1, node2) += interactions(node1, node2);
          out_interactions(node1, node2) /= iteration + 1;
          out_interactions(node2, node1) = out_interactions(node1, node2);
        }

        if(variable_type[node1] == "ordinal") {
          for(int category = 0; category < no_categories[node1]; category++) {
            out_thresholds(node1, category) *= iteration;
            out_thresholds(node1, category) += thresholds(node1, category);
            out_thresholds(node1, category) /= iteration + 1;
          }
        }
        if(variable_type[node1] == "ordinal_BC") {
          out_thresholds(node1, 0) *= iteration;
          out_thresholds(node1, 0) += thresholds(node1, 0);
          out_thresholds(node1, 0) /= iteration + 1;
          out_thresholds(node1, 1) *= iteration;
          out_thresholds(node1, 1) += thresholds(node1, 1);
          out_thresholds(node1, 1) /= iteration + 1;
        }
      }
      if(variable_type[no_nodes - 1] == "ordinal") {
        for(int category = 0; category < no_categories[no_nodes - 1]; category++) {
          out_thresholds(no_nodes - 1, category) *= iteration;
          out_thresholds(no_nodes - 1, category) += thresholds(no_nodes - 1, category);
          out_thresholds(no_nodes - 1, category) /= iteration + 1;
        }
      }
      if(variable_type[no_nodes - 1] == "ordinal_BC") {
        out_thresholds(no_nodes - 1, 0) *= iteration;
        out_thresholds(no_nodes - 1, 0) += thresholds(no_nodes - 1, 0);
        out_thresholds(no_nodes - 1, 0) /= iteration + 1;
        out_thresholds(no_nodes - 1, 1) *= iteration;
        out_thresholds(no_nodes - 1, 1) += thresholds(no_nodes - 1, 1);
        out_thresholds(no_nodes - 1, 1) /= iteration + 1;
      }
    }
  }


  if(edge_selection == true) {
    return List::create(Named("gamma") = out_gamma,
                        Named("interactions") = out_interactions,
                        Named("thresholds") = out_thresholds);
  } else {
    return List::create(Named("interactions") = out_interactions,
                        Named("thresholds") = out_thresholds);
  }
}