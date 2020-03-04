function model = init_dp(model)
    %for variational posteriors/statistics    
    nmod = model.nmod;
    nvec = model.nvec;
    if ~isfield(model.dp, 'T')
        model.dp.T = round(nvec/10);    
    end
    if ~isfield(model.dp, 'lam')
        model.dp.lam = 1.0*ones(nmod,1);
    else
        %take inverse for convenience
        model.dp.lam= 1./model.dp.lam;
    end
    if ~isfield(model.dp, 's')
        model.dp.s = 1.0*ones(nmod,1);
    else
        %take inverse for convenience
        model.dp.s = 1./model.dp.s;
    end
    if ~isfield(model.dp, 'var_iter')
        model.dp.var_iter = 50;
    end
    model.dp.alpha = 1.0;
    model.dp.ga = cell(nmod,1);
    model.dp.stat = [];

    %expectation of log(v) and log(1-v)
    model.dp.stat.ex_logv = cell(nmod,1);
    %parameters for cluster membership
    model.dp.stat.phi = cell(nmod,1);
    %parameters for cluster centers
    model.dp.stat.eta_mean = cell(nmod,1);
    model.dp.stat.eta_cov = cell(nmod,1);
    %expt for norm^2
    model.dp.stat.eta_norm2 = cell(nmod,1);

    for k=1:nmod
        trunc_no = model.dp.T(k);
        model.dp.ga{k} = zeros(trunc_no, 2);
        model.dp.stat.ex_logv{k} = zeros(trunc_no, 2);
        %model.dp.stat.phi{k} = drchrnd(ones(1, trunc_no),nvec(k));
        model.dp.stat.phi{k} = 1/trunc_no*ones(nvec(k), trunc_no);
        model.dp.stat.eta_mean{k} = rand(trunc_no, model.dim(k));
        %model.dp.stat.eta_mean{k} = randn(trunc_no, model.dim(k));
        model.dp.stat.eta_cov{k} = 1/model.dp.lam(k)*ones(1, trunc_no);
        model.dp.stat.eta_norm2{k} = model.dim(k)*model.dp.stat.eta_cov{k} + sum(model.dp.stat.eta_mean{k}.^2,2)';
    end
    %create the cache stats
    model.dp.cache_stat = [];
    %\sum \Phi_k^{n,t}
    model.dp.cache_stat.psy1 = cell(nmod,1);
    %\sum \Phi_k^{n,t}U_k^n
    model.dp.cache_stat.psy2 = cell(nmod,1);
    %\sum_{n=1}^m_k \sum_{j=t+1}^T \Phi_k^{n,j}
    model.dp.cache_stat.psy3 = cell(nmod,1);
    for k=1:nmod
        model.dp.cache_stat.psy1{k} = sum(model.dp.stat.phi{k}, 1);
        model.dp.cache_stat.psy2{k} = model.dp.stat.phi{k}'*model.U{k};
        model.dp.cache_stat.psy3{k} = zeros(1, model.dp.T(k));
        csum = cumsum(fliplr(model.dp.cache_stat.psy1{k}));
        csum = fliplr(csum(1:end-1));
        model.dp.cache_stat.psy3{k} = [csum,0];                                
    end 
end