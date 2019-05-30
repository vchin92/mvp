function output = gibbs_HMC_beta(n_obs, n_dim, n_pred, y_cts, X, ...
    corr_mat, prior_beta, old_beta)

B0 = prior_beta.mat;
b0 = prior_beta.mean;
 
inv_corr_mat = inv(corr_mat);
xx           = X' * X;
xy           = repelem(X, n_dim, 1) .* reshape(y_cts', n_obs * n_dim, 1);
xy           = reshape(xy, [n_dim, n_obs, n_pred]);
xy           = squeeze(sum(xy, 2));

post_cov_mat = repelem(inv_corr_mat, n_pred, n_pred) .* repmat(xx, ...
               n_dim, n_dim);
post_cov_mat = inv(post_cov_mat + inv(B0));
post_mean    = repmat(xy, 1, n_dim) .* repelem(inv_corr_mat, 1, n_pred);
post_mean    = post_cov_mat * (sum(post_mean)' + B0 \ b0); %#ok<MINV>

output = 2 * post_mean - old_beta;
end 