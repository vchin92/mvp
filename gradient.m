function output = gradient(error, L)

% Gradients of potential energy

[n_obs, dim_y] = size(error);
l              = zeros(dim_y, dim_y);
lower_tri      = triu(true(dim_y, dim_y), 1)';
l(lower_tri)   = L;
l              = l + eye(dim_y);
sum_sq         = sum(l.^2, 2);
sum_sq         = repmat(sum_sq, 1, dim_y);
denominator    = sum_sq(lower_tri)';
error_sq       = error' * error;

[d_log_detR, d_invR, d_log_detsubmat] = derivative(L, dim_y);

output = zeros(1, length(L));

for i=1:length(L)
    output(i) = - 1/2 * sum(sum(d_invR{1,i} .* error_sq'));
end

output = (- n_obs/2 + 1/2 * dim_y * (dim_y - 1) - 1) * d_log_detR + ...
         output - (dim_y + 1)/2 * d_log_detsubmat;
     
% Jacobian
output = - output + (dim_y + 1) * L ./ denominator;
end