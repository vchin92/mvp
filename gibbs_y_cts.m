function output = gibbs_y_cts(n_dim, y_cts, reff, XB, corr_mat, limit)

% Gibbs sampling from conditional posterior distribution of latent
% variables y^* (truncated multivariate normal distribution)

lower_lim   = limit.lower_limit;
upper_lim   = limit.upper_limit;
uncond_mean = XB + reff;

for i=1:n_dim    
    cf         = corr_mat(i, [1:(i-1) (i+1):end]) / ...
                 corr_mat([1:(i-1) (i+1):end], [1:(i-1) (i+1):end]);
    cond_mean  = uncond_mean(:,i) + (y_cts(:, [1:(i-1) (i+1):end]) - ...
                 uncond_mean(:, [1:(i-1) (i+1):end])) * cf';
    cond_var   = corr_mat(i,i) - cf * corr_mat([1:(i-1) (i+1):end], i);
    cond_sd    = sqrt(cond_var);
    y_cts(:,i) = trandn((lower_lim(:,i) - cond_mean) / cond_sd, ...
                 (upper_lim(:,i) - cond_mean) / cond_sd) * cond_sd + ...
                 cond_mean;
end
output = y_cts;
end 