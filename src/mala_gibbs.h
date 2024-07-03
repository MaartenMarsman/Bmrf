#include <Rcpp.h>
using namespace Rcpp;

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
                                        double epsilon_hi);

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
                       double epsilon_hi);


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
                                IntegerVector reference_category);