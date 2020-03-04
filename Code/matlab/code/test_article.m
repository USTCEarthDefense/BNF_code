clear all;
close all;

addpath_recurse('./util');
addpath_recurse('./tensor_toolbox_2.6');
addpath_recurse('./lightspeed');
addpath_recurse('./minFunc_2012');

rng('default');

load('../data/article-train-hybrid-3-300.mat')
data.e = data.e';
model = [];
model.R = 8;
nvec = max(data.ind);
nmod = size(nvec,2);
for k=1:nmod
    model.U{k} = rand(nvec(k), model.R);
    %model.U{k} = 0.1*randn(n,R);
end
model.oldU = model.U;
model.lam = 0.01;
model.nepoch = 50;
model.dim = model.R*ones(1,nmod);
model.np = 100;
model.decay = 0.97;
model.batch_size = 100;
model.a = 0.1;
model.b = 0.1;
model.a0 = 1e-3;
model.a1 = 1e-3;
model.b0 = 1e-3;
model.b1 = 1e-3;
model.tau = 1;
model.T = data.T;
model.init_opt = 'random';

model.Dmax = 1;%maximum one hour

%load testing data
test = load('../data/article-test-hybrid-3-300-all-remaining.mat');
test = test.data;
test.e = test.e';

tic,
[model, test_LL_approx, test_LL_ELBO, models] = online_inference_TensorHPGP_doubly_sgd_dp_v2(data, model, 1, test, (1:length(test.e))');
toc
res = test_ELBO(model,test,(1:length(test.e))');
fprintf('test ELBO = %g\n', res);

res = test_ll_approx(model,test,(1:length(test.e))');
fprintf('test approx. LL = %g\n', res);

res = [];
res.epoch = 1:model.nepoch;
res.ELBO = test_LL_ELBO;
res.LL_approx = test_LL_approx;
save('rfp-hp-article-all-remaning-R8-3-300.mat', 'res');
save('rfp-hp-article-models-all_rermaning.mat', 'models');


