#include <Rcpp.h>
using namespace Rcpp;

/// ----------------------------------------------------------------------------|
// The log pseudolikelihood ratio [proposed against current] for an interaction
// ----------------------------------------------------------------------------|
double log_pseudolikelihood_ratio(NumericMatrix interactions,
                                  NumericMatrix thresholds,
                                  IntegerMatrix observations,
                                  IntegerVector no_categories,
                                  int no_persons,
                                  int variable1,
                                  int variable2,
                                  double proposed_state,
                                  double current_state,
                                  NumericMatrix rest_matrix,
                                  LogicalVector variable_bool,
                                  IntegerVector reference_category);