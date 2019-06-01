% Loading data
load('simulated_data.mat')

n_dim      = size(data.outcome, 2); % dimension of outcome variables
n_mcmc     = 30000; % number of MCMC iterations (including burnin)
burnin     = 5000;  % number of burnin iterations
n_pred     = size(data.covariate, 2) * n_dim; % number of predictors
antithetic = 0; % 0 for independent sampling; 1 for antithetic sampling
n_da       = 4; % number of dual averaging HMC stepsize adaptation during
                % burnin

% Prior distributions for regression coefficients and covariance matrix of
% random effects:
% beta ~ N(0, 100I)
% Sigma_reff ~ HIW(2, 1)
prior.prior_beta.mean      = zeros(n_pred, 1);
prior.prior_beta.mat       = 100 * eye(n_pred);
prior.prior_rand_cov.scale = ones(1,n_dim);
prior.prior_rand_cov.df    = 2; % So that implied marginal correlation 
                                % coefficients is uniform on [-1,1].

% Upper and lower limits of latent variables underlying discrete outcomes.
lower_lim             = zeros(size(data.outcome));
upper_lim             = zeros(size(data.outcome));
logic_zero            = data.outcome < 1;
logic_one             = data.outcome > 0;
lower_lim(logic_zero) = -Inf;
upper_lim(logic_one)  = Inf;
limit.lower_limit     = lower_lim;
limit.upper_limit     = upper_lim;

clearvars -except data n_mcmc burnin limit prior n_da antithetic

[y_cts, reff, rand_cov, reg_coef, corr_mat, chol_R] = mvp(data, n_mcmc, ...
    burnin, limit, prior, n_da, antithetic);