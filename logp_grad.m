function [logp, grad] = logp_grad(chol_R, error)
chol_R = chol_R';
grad   = gradient(error, chol_R)';
logp   = - potential(error, chol_R);
end