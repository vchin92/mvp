function output = HMC(epsilon, leapfrog, y_cts, XB, random_eff, current_q)

% Hamiltonian Monte Carlo

error     = y_cts - random_eff - XB;
q         = current_q;
p         = randn(1, length(q));
current_p = p;
p         = p - epsilon * gradient(error, q) / 2;

for i=1:leapfrog
    q = q + epsilon * p;
    if i~=leapfrog
        p = p - epsilon * gradient(error, q);
    end
end

p = p - epsilon * gradient(error, q) / 2;
p = -p;

current_U  = potential(error, current_q);
current_K  = sum(current_p.^2) / 2;
proposed_U = potential(error, q);
proposed_K = sum(p.^2) / 2;

if rand(1) < exp(current_U - proposed_U + current_K - proposed_K)
    output = q;
else
    output = current_q;
end
end