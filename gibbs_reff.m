function [output, inv_cov_mat] = gibbs_reff(n_ppl, n_time, n_dim, ...
    y_cts, XB, corr_mat, reff_cov_mat)

% Gibbs sampling from conditional posterior distribution of zero-mean
% random effects using INDEPENDENT sampling.

inv_corr_mat = inv(corr_mat);
inv_cov_mat  = inv(reff_cov_mat);
diff         = y_cts - XB;

precision  = n_time .* inv_corr_mat + inv_cov_mat;
covariance = inv(precision);
mean_indv  = reshape(sum(reshape(diff, n_time, [])), [], size(diff, 2));
mean_indv  = (covariance * (inv_corr_mat * mean_indv'))'; %#ok<MINV>
cholesky   = chol(covariance, 'upper');

output = mvnrnd(zeros(1, n_dim), eye(n_dim), n_ppl) * cholesky + ...
         mean_indv;
end 