% Added seed
rng(1);

bias = 1; scaling = 10; sparsity = 10; solutionSparsity = 0.1;

% Experiment 1 (L2-regularized sparse least squares)
n = 1000; p = 1000;
X = randn(n,p)+bias;
X = X*diag(scaling*randn(p,1));
X = X .* (rand(n,p) < sparsity*log(n)/n);
w = randn(p,1);
y = X*w + randn(n,1);
save('exp1.mat','X','y');

% Experiment 2 (L2-regularized sparse logistic regression)
y = sign(X*w);
y = y .* sign(rand(n,1)-.1);
save('exp2.mat','X','y');

% Experiment 3 (Over-determined dense least squares)
n = 1000;
p = 100;
X = randn(n,p)+bias;
X = X*diag(scaling*randn(p,1));
w = randn(p,1);
y = X*w + randn(n,1);
save('exp3.mat','X','y');

% Experiment 4 (L1-regularized under-determined sparse least squares)
n = 1000;
p = 10000;
X = randn(n,p)+bias;
X = X*diag(scaling*randn(p,1));
X = X .* (rand(n,p) < sparsity*log(n)/n);
w = randn(p,1) .* (rand(p,1) < solutionSparsity);
y = X*w + randn(n,1);
save('exp4.mat','X','y');
