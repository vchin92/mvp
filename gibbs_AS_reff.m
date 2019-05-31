function [output, inv_cov_mat] = gibbs_AS_reff(n_time, y_cts, XB, ...
    corr_mat, reff_cov_mat, old_reff)

% Gibbs sampling from conditional posterior distribution of zero-mean
% random effects using ANTITHETIC sampling.

inv_corr_mat = inv(corr_mat);
inv_cov_mat  = inv(reff_cov_mat);
diff         = y_cts - XB;

precision  = n_time .* inv_corr_mat + inv_cov_mat;
covariance = inv(precision);
mean_indv  = reshape(sum(reshape(diff, n_time, [])), [], size(diff, 2));
mean_indv  = (covariance * (inv_corr_mat * mean_indv'))'; %#ok<MINV>

output = 2 * mean_indv - old_reff;
end 