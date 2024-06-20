//////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////// Rcpp code for the "composte quantile regression  /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>
#include <algorithm>

using namespace Rcpp;
using namespace arma;

// Inverse Gaussian sampler
double rinvGauss(double mu, double lambda) {
  double nu = R::rnorm(0, 1);
  double y = nu * nu;
  double x = mu + (mu * mu * y) / (2.0 * lambda) - (mu / (2.0 * lambda)) * sqrt(4.0 * mu * lambda * y + mu * mu * y * y);
  double z = R::runif(0, 1);
  if (z <= mu / (mu + x)) {
    return x;
  } else {
    return (mu * mu) / x;
  }
}

// Dirichlet sampler
arma::vec rdirichlet(int K, arma::vec alpha) {
  arma::vec samples(K);
  for (int k = 0; k < K; ++k) {
    samples(k) = R::rgamma(alpha(k), 1.0);
  }
  return samples / sum(samples);
}

// Multinomial sampler
arma::uvec rmultinom_custom(int n, int size, const arma::vec& prob) {
  arma::uvec result(n);
  Environment stats("package:stats"); // Access the 'stats' namespace
  Function rmultinom = stats["rmultinom"]; // Get the 'rmultinom' function from 'stats'
  for (int i = 0; i < n; ++i) {
    NumericMatrix temp = rmultinom(1, size, Rcpp::wrap(prob)); // Call rmultinom
    NumericVector temp_col = temp(_, 0); // Extract the column as a NumericVector
    result(i) = std::distance(temp_col.begin(), std::max_element(temp_col.begin(), temp_col.end())) + 1; // Find max index and convert to 1-based index
  }
  return result;
}

// Custom function to simulate multinomial distribution and find the index of the maximum value
int rmultinom_custom2(const arma::vec& prob) {
  Environment stats("package:stats"); // Access the 'stats' namespace
  Function rmultinom = stats["rmultinom"]; // Get the 'rmultinom' function from 'stats'
  NumericMatrix temp = rmultinom(1, 1, Rcpp::wrap(prob)); // Call rmultinom with size = 1
  NumericVector temp_col = temp(_, 0); // Extract the column as a NumericVector
  return std::distance(temp_col.begin(), std::max_element(temp_col.begin(), temp_col.end())); // Find max index (0-based)
}


// [[Rcpp::export]]
arma::uvec generate_subsample_idx(int n_sampler, int n_burn, int thin) {
  std::vector<int> temp;
  for (int i = 1; i <= (n_sampler - n_burn); i += thin) {
    temp.push_back(i);
  }
  arma::uvec subsample_idx(temp.size());
  for (size_t i = 0; i < temp.size(); ++i) {
    subsample_idx(i) = temp[i] + n_burn - 1;  // Adjust for zero-based index in C++
  }
  return subsample_idx;
}


// [[Rcpp::export]]
Rcpp::List cqr_lasso(const arma::mat& x, const arma::vec& y, int K = 9, int n_sampler = 13000, int n_burn = 3000, int thin = 20) {
  // Initialize variables
  vec theta = linspace(1.0, K, K) / (K + 1.0);
  int n = x.n_rows;
  int p = x.n_cols;
  vec xi1 = (1 - 2 * theta) / (theta % (1 - theta));
  vec xi2 = sqrt(2 / (theta % (1 - theta)));
  vec eps = y - x * solve(x.t() * x, x.t() * y);

  // Priors
  double a = 1e-1;
  double b = 1e-1;
  double c = 1e-1;
  double d = 1e2;

  // Initialization
  vec alpha_c = ones(K);
  vec pi_init = alpha_c / sum(alpha_c);
  uvec zi_c = rmultinom_custom(n,1, pi_init);
  vec xi1_c = xi1.elem(conv_to<uvec>::from(zi_c - 1));
  vec xi2_c = xi2.elem(conv_to<uvec>::from(zi_c - 1));
  vec pi_c = ones(K) / K;
  vec beta_c = ones(p);
  vec tz_c = ones(n);
  vec s_c = ones(p);
  double tau_c = 1;
  double eta2_c = 1;
  vec b_k = quantile(eps, theta);
  vec b_c = b_k.elem(conv_to<uvec>::from(zi_c - 1));


  // Iteration
  mat zi_p = zeros(n_sampler, n);
  mat pi_p = ones(n_sampler, K) / K;
  mat beta_p = zeros(n_sampler, p);
  mat tz_p = zeros(n_sampler, n);
  mat b_p = zeros(n_sampler, K);
  vec tau_p = zeros(n_sampler);
  vec eta2_p = zeros(n_sampler);
 // vec dic_p = zeros(n_sampler);

  for (int iter = 0; iter < n_sampler; ++iter) {
    if (iter % 1000 == 0) {
      Rcpp::Rcout << "This is step " << iter << std::endl;
    }

    // Full conditional for tz (nu)
    vec temp_lambda = pow(xi1_c,2) * tau_c / pow(xi2_c,2) + 2 * tau_c;
    vec temp_nu = sqrt(temp_lambda % pow(xi2_c,2) / (tau_c * pow(y - b_c - x * beta_c,2)));
    for (int i = 0; i < n; ++i) {
      tz_c(i) = 1 / rinvGauss(temp_lambda(i), temp_nu(i));
    }

    // Full conditional for s
    temp_lambda.fill(eta2_c);
    for (int j = 0; j < p; ++j) {
      s_c(j) = 1 / rinvGauss(temp_lambda(j), sqrt(temp_lambda(j) / pow(beta_c(j),2)));
    }

    // Full conditional for beta
    for (int k = 0; k < p; ++k) {
      double temp_var = 1 / (sum(pow(x.col(k),2) * tau_c / (pow(xi2_c,2) % tz_c)) + 1 / s_c(k));
      double temp_mean = sum(x.col(k) % (y - b_c - xi1_c % tz_c - x * beta_c + x.col(k) * beta_c(k)) * tau_c / (pow(xi2_c,2) % tz_c)) * temp_var;
      beta_c(k) = R::rnorm(temp_mean, sqrt(temp_var));
    }

    // Full conditional for tau
    double temp_shape = a + 1.5 * n;
    double temp_rate = sum(pow(y - b_c - x * beta_c - xi1_c % tz_c,2) / (2 * pow(xi2_c,2) % tz_c) + tz_c) + b;
    tau_c = R::rgamma(temp_shape, 1 / temp_rate);

    // Full conditional for eta2
    temp_shape = p + c;
    temp_rate = sum(s_c) / 2 + d;
    eta2_c = 1; // R::rgamma(temp_shape, 1 / temp_rate);

    // Full conditional for zi (c)
    for (int i = 0; i < n; ++i) {
      vec temp_power = pow(y(i) - b_k - sum(x.row(i) * beta_c) - xi1 * tz_c(i),2) * tau_c / (pow(xi2,2) * tz_c(i));
      vec temp_alpha = pi_c % exp(-0.5 * temp_power) / xi2;
      vec norm_alpha = temp_alpha / sum(temp_alpha);
      zi_c(i) =  rmultinom_custom2(norm_alpha) +1 ;
    }


    // Debug print for checking zi_c
  //  Rcpp::Rcout << "zi_c at iteration " << iter << ": " << zi_c.t() << std::endl;

    // Debug print for checking zi_c
 //   Rcpp::Rcout << "xi1_c at iteration " << iter << ": " << xi1_c.t() << std::endl;

    // Debug print for checking zi_c
  //  Rcpp::Rcout << "xi2_c at iteration " << iter << ": " << xi2_c.t() << std::endl;



     xi1_c = xi1.elem(conv_to<uvec>::from(zi_c -1 ));
     xi2_c = xi2.elem(conv_to<uvec>::from(zi_c -1 ));


// Debug print for checking zi_c
// Rcpp::Rcout << "xi1_c at iteration " << iter << ": " << xi1_c.t() << std::endl;
// Rcpp::Rcout << "xi2_c at iteration " << iter << ": " << xi2_c.t() << std::endl;

    // Full conditional for pi (omega)
    vec n_c(K, fill::zeros);

    for (int i = 0; i < n; ++i) {
      ++n_c(zi_c(i) -1 );  // Decrement zi_c(i) by 1 to match the zero-based indexing of n_c
    }

    pi_c = rdirichlet(K, n_c + alpha_c);


    // Full conditional for b
    //double dic_c = 0;
    for (int k = 0; k < K; ++k) {
      uvec which_k = find(zi_c == (k +1 ));


      if (!which_k.is_empty()) {
        vec bc = y.elem(which_k) - x.rows(which_k) * beta_c - xi1_c.elem(which_k) % tz_c.elem(which_k);
        vec sc = tau_c / (pow(xi2_c.elem(which_k),2) % tz_c.elem(which_k));
        double mean = sum(bc % sc) / sum(sc);
        double sd = 1 / sqrt(sum(sc));

        // Debug: print the mean and sd before generating b_k
      //  Rcpp::Rcout << "k = " << k << ", mean = " << mean << ", sd = " << sd << std::endl;

        b_k(k) = R::rnorm(mean, sd);
        vec uu = y.elem(which_k) - b_k(k) - x.rows(which_k) * beta_c;

        uvec pos_indices = find(uu >= 0);
        uu.elem(pos_indices) = theta(k) * uu.elem(pos_indices);

        uvec neg_indices = find(uu < 0);
        uu.elem(neg_indices) = (theta(k) - 1) * uu.elem(neg_indices);

      }
    }
    b_c = b_k.elem(conv_to<uvec>::from(zi_c  -1));

    // Store samples
    zi_p.row(iter) = conv_to<rowvec>::from(zi_c);
    beta_p.row(iter) = beta_c.t();
    tau_p(iter) = tau_c;
    eta2_p(iter) = eta2_c;
    pi_p.row(iter) = pi_c.t();
    b_p.row(iter) = b_k.t();
  }

  // Generate subsample index 
  arma::uvec subsample_idx = generate_subsample_idx(n_sampler, n_burn, thin);
  
  arma::mat beta_save = beta_p.rows(subsample_idx);
  arma::vec tau_save = tau_p.elem(subsample_idx);
  arma::vec eta2_save = eta2_p.elem(subsample_idx);
  arma::mat pi_save = pi_p.rows(subsample_idx);
  arma::mat b_save = b_p.rows(subsample_idx);
  
  List result;
  result["beta"] = beta_save;
  result["tau"] = tau_save;
  result["eta2"] = eta2_save;
  result["pi"] = pi_save;
  result["b"] = b_save;
 return result;


}
