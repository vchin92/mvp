function output = potential(error, L)

% Potential energy component of Hamiltonian (i.e. negative log posterior)

[n_obs, dim_y] = size(error);
l              = zeros(dim_y, dim_y);
lower_tri      = triu(true(dim_y, dim_y), 1)';
l(lower_tri)   = L;
l              = l + eye(dim_y);
correlation    = corrcov(l * l');

tmp = 0;

for i=1:dim_y
    tmp = tmp + log(det(correlation([1:(i-1) (i+1):end], ...
          [1:(i-1) (i+1):end])));
end

log_posterior = (- n_obs / 2 + 1/2 * dim_y * (dim_y - 1) - 1) * ...
                log(det(correlation)) - 1/2 * sum(sum(inv(correlation) ...
                .* (error' * error)')) - (dim_y + 1)/2 * tmp; 

% Jacobian
output = - log_posterior + (dim_y + 1)/2 * (sum(log(sum(l.^2,2))));
end