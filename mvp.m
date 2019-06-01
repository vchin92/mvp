function [y_cts, reff, rand_cov, reg_coef, corr_mat, chol_R] = ...
    mvp(data, n_mcmc, burnin, limit, prior, n_da, antithetic)

% Args:
% data       - structure containing outcome variables, covariates and
%              number of individuals in dataset (assuming BALANCED panel)
% n_mcmc     - number of MCMC iterations
% burnin     - number of MCMC burnins
% limit      - structure containing upper and lower limits of latent
%              variables underlying discrete outcomes based on the data
%              augmentation representation of Albert and Chib (1993)
% prior      - structure containing prior distributions for parameters
% n_da       - number of dual averaging HMC stepsize adaptation during
%              burnin
% antithetic - 0 for independent sampling; 1 for antithetic sampling

% Returns:
% Let N be total number of observations.
%     P be number of individual.
%     D be number of outcome dimension.
%     T be number of MCMC iterations including burnin.
%     K be total number of regression coefficients.
% y_cts    - latent variables underlying discrete outcome. Array of size
%            (N * D * T).
% reff     - zero-mean random effects. Array of size (P * D * T).
% rand_cov - covariance matrix of random effects. Array of size (D * D *T).
% reg_coef - regression coefficients. Array of size (K * T) with the first
%            K/D values of each column representing regression cofficients
%            for the first margin of outcome variables and so on.
% corr_mat - correlation matrix of errors. Array of size (D * D * T).
% chol_R   - Cholesky factor for correlation matrix. Array of size
%            ((D^2-D)/2 * T).

% Copyright (c) 2019, Vincent Chin
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:
% 
% Redistributions of source code must retain the above copyright notice, 
% this list of conditions and the following disclaimer.
% Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
% IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

[n_obs, n_dim] = size(data.outcome);
n_ppl          = data.num_ppl;
n_time         = n_obs / n_ppl;
n_pred         = size(data.covariate, 2) * n_dim;
lower_tri      = triu(true(n_dim, n_dim), 1)';

y_cts    = zeros(n_obs, n_dim, n_mcmc);
reff     = zeros(n_ppl, n_dim, n_mcmc);
rand_cov = zeros(n_dim, n_dim, n_mcmc);
reg_coef = zeros(n_pred, n_mcmc);
corr_mat = zeros(n_dim, n_dim, n_mcmc);
chol_R   = zeros((n_dim^2 - n_dim)/2, n_mcmc);

X  = data.covariate;
XB = X * reshape(reg_coef(:,1), n_pred / n_dim, n_dim);

identity   = eye(n_dim);
zero_mat   = zeros(n_dim, n_dim);
shape      = (prior.prior_rand_cov.df + n_dim) / 2;
inv_A_sq   = 1 ./ (prior.prior_rand_cov.scale.^2);

% Horseshoe prior on regression coefficients (non-intercept terms)
interc               = reshape(1:n_pred, n_pred / n_dim, n_dim);
non_interc           = interc(2:end,:);
non_interc           = reshape(non_interc, 1, n_pred - n_dim);
interc               = interc(1,:);
p0                   = length(non_interc);
v                    = 1 ./ gamrnd(1/2, 1, 1, p0);
xi                   = 1 ./ gamrnd(1/2, 1);
tau_sq               = 1 ./ gamrnd(1/2, xi);
lambda_sq            = 1 ./ gamrnd(1/2, v);
psi                  = lambda_sq * tau_sq;
psi_zero             = zeros(1, n_pred);
psi_diag             = psi_zero;
psi_diag(interc)     = prior.prior_beta.mat(1,1);
psi_diag(non_interc) = psi;
prior.prior_beta.mat = diag(psi_diag);

% Initialised values for parameters
y_cts(:,:,1)        = abs(randn(n_obs, n_dim)) .* (2 * data.outcome - 1);
reff(:,:,1)         = randn(n_ppl, n_dim);
rand_cov(:,:,1)     = iwishrnd(eye(n_dim), n_dim + 1);
reg_coef(:,1)       = randn(n_pred, 1);
chol_R(:,1)         = randn((n_dim^2 - n_dim)/2, 1);
chol_mat            = zero_mat;
chol_mat(lower_tri) = chol_R(:,1);
chol_mat            = chol_mat + identity;
corr_mat(:,:,1)     = corrcov(chol_mat * chol_mat');
full_reff           = repelem(reff(:,:,1), n_time, 1);

% Choose suitable HMC stepsize based on initialised values
error              = y_cts(:,:,1) - full_reff - XB;
f                  = @(chol_R) logp_grad(chol_R, error);
[~, epsilon, ~, ~] = dualAveraging(f, chol_R(:,1));

for i=2:n_mcmc
    
    %% Updating latent variables y^*
    y_cts(:,:,i) = gibbs_y_cts(n_dim, y_cts(:,:,i-1), full_reff, XB, ...
                   corr_mat(:,:,i-1), limit);
    
    %% Updating regression coefficients using independent sampling (before
    %% burnin) and antithetic sampling (after burnin)
    if and(i > burnin, antithetic == 1)
        reg_coef(:,i) = gibbs_AS_beta(n_obs, n_dim, n_pred / n_dim, ...
                        y_cts(:,:,i) - full_reff, X, corr_mat(:,:,i-1), ...
                        prior.prior_beta, reg_coef(:,i-1));
    else
        reg_coef(:,i) = gibbs_beta(n_obs, n_dim, n_pred / n_dim, ...
                        y_cts(:,:,i) - full_reff, X, corr_mat(:,:,i-1), ...
                        prior.prior_beta);
    end
               
    %% Updating Cholesky factors using HMC No-U-Turn Sampler
    B     = reshape(reg_coef(:,i), n_pred/n_dim, n_dim);   
    XB    = X * B;
    error = y_cts(:,:,i) - full_reff - XB;
    f     = @(chol_R) logp_grad(chol_R, error);
    
    % Adapt leapfrog stepsize using dual averaging method
    if any((1:n_da) * ceil(burnin/(n_da+1)) == i)
        [chol_R(:,i), epsilon, ~, ~] = dualAveraging(f, chol_R(:,i-1));
    else
        chol_R(:,i) = NUTS(f, epsilon, chol_R(:,i-1));
    end
                  
    %% Resulting correlation matrix after updating Cholesky factors      
    chol_mat            = zero_mat;
    chol_mat(lower_tri) = chol_R(:,i);
    chol_mat            = chol_mat + identity;
    corr_mat(:,:,i)     = corrcov(chol_mat * chol_mat');
    
    %% Updating random effects using independent sampling (before burnin)
    %% and antithetic sampling (after burnin)
    if and(i > burnin, antithetic == 1)
        [tmp_reff, inv_rand_cov] = gibbs_AS_reff(n_time, y_cts(:,:,i), ...
                                   XB, corr_mat(:,:,i), ...
                                   rand_cov(:,:,i-1), reff(:,:,i-1));
    else
        [tmp_reff, inv_rand_cov] = gibbs_reff(n_ppl, n_time, n_dim, ...
                                   y_cts(:,:,i), XB, corr_mat(:,:,i), ...
                                   rand_cov(:,:,i-1));
    end
    reff(:,:,i) = tmp_reff;
    
    %% Updating covariance matrix of random effects based on hierarchical
    %% inverse-Wishart prior distribution of Huang and Wand (2013)
    rate            = prior.prior_rand_cov.df * diag(inv_rand_cov)' + ...
                      inv_A_sq;
    a               = 1 ./ gamrnd(shape, 1 ./ rate);
    rand_cov(:,:,i) = iwishrnd(tmp_reff' * tmp_reff + 2 * ...
                      prior.prior_rand_cov.df * diag(1./a), n_ppl + 2 * ...
                      shape - 1);
                  
    full_reff = repelem(tmp_reff, n_time, 1);
    
    %% Updating parameters of horseshoe prior using the latent variable
    %% representation of Makalic and Schmidt (2016)
    v         = 1 ./ gamrnd(1, 1 ./ (1 + 1 ./ lambda_sq));
    xi        = 1 ./ gamrnd(1, 1 ./ (1 + 1 / tau_sq));
    bj        = reg_coef(non_interc, i)';
    bj2       = bj.^2;
    tau_sq    = 1 ./ gamrnd((p0 + 1) / 2, 1 / (1 / xi + 1 / 2 * ...
                sum(bj2 ./ lambda_sq)));
    lambda_sq = 1 ./ gamrnd(1, 1 ./ (1 ./ v + bj2 / (2 * tau_sq)));
    
    psi                  = lambda_sq * tau_sq;
    psi_diag             = psi_zero;
    psi_diag(interc)     = prior.prior_beta.mat(1,1);
    psi_diag(non_interc) = psi;
    prior.prior_beta.mat = diag(psi_diag);
    
    
    
    if mod(i, 1000) == 0
        fprintf('%d iterations have been completed.\n', i);
        subplot(1,4,1)
        plot(reg_coef(1,1:i));
        subplot(1,4,2)
        plot(reg_coef(20,1:i));
        subplot(1,4,3)
        plot(chol_R(8,1:i));
        subplot(1,4,4)
        plot(chol_R(end,1:i));
        drawnow
    end
end
end