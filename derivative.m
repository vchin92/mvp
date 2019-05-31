function [d_log_detR, d_invR, d_log_detsubmat] = derivative(L, dim_y)

% Derivatives required for the computation of the gradient information in
% Hamiltonian Monte Carlo algorithm

d_log_detR      = zeros(1, length(L));
d_invR          = cell(1, length(L));
d_log_detsubmat = zeros(1, length(L));

max_i = dim_y;
max_j = max_i - 1;
index = 1;

lower_tri    = triu(true(dim_y, dim_y),1)';
l            = zeros(dim_y, dim_y);
l(lower_tri) = L;
l            = l + eye(dim_y);
Sigma        = l * l';
invSigma     = inv(Sigma);
R            = corrcov(Sigma);
invR         = inv(R);

diag_Sigma             = diag(Sigma);
diag_Sigma_half        = diag_Sigma.^(1/2);
diag_Sigma_minushalf   = diag_Sigma.^(-1/2);
diag_Sigma_minus3over2 = diag_Sigma.^(-3/2);

inv_submat = cell(1, dim_y);
for i=1:dim_y
    inv_submat{1,i} = inv(R([1:(i-1) (i+1):end], [1:(i-1) (i+1):end]));
end

for j=1:max_j
    for i=(j+1):max_i        
        
        %% Derivative of inverse correlation matrix wrt Cholesky factors
        
        deriv      = zeros(dim_y, dim_y);
        deriv(i,:) = l(:,j);
        dSigma_Lij = deriv + deriv';
        
        tmp6           = invSigma * dSigma_Lij;
        dinv_Sigma_Lij = - tmp6 * invSigma;
        
        d_diagSigma_minushalf_Lij = -1/2 * diag_Sigma_minus3over2 ...
                                    .* diag(dSigma_Lij);
        
        tmp4 = (d_diagSigma_minushalf_Lij * diag_Sigma_minushalf') .* invR;
        
        output          = (diag_Sigma_half * diag_Sigma_half') .* ...
                          (dinv_Sigma_Lij - tmp4 - tmp4');
        d_invR{1,index} = output;
        
        %% Derivative of log determinant of submatrix of correlation matrix
        %% wrt Cholesky factors
        
        tmp3 = 0;
        tmp5 = (d_diagSigma_minushalf_Lij * diag_Sigma_minushalf').* Sigma;
        tmp7 = (tmp5 + (diag_Sigma_minushalf * diag_Sigma_minushalf') ...
               .* dSigma_Lij + tmp5');
           
        for k=1:dim_y
            tmp3 = tmp3 + sum(sum(inv_submat{1,k} .* tmp7([1:(k-1) ...
                   (k+1):end], [1:(k-1) (k+1):end])'));
        end
        d_log_detsubmat(index) = tmp3;
        
        %% Derivative of log determinant of correlation matrix wrt Cholesky
        %% factors
        
        tmp1              = - 2 * l(i,j) / sum(l(i,:).^2);
        tmp2              = trace(tmp6);
        d_log_detR(index) = tmp1 + tmp2;
        index             = index + 1;
    end
end
end