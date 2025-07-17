clc,clear
%% cat
X=im2double(imread('cat1.jpg'));
sz = size(X);
ranks = [4 4 4];
ranks_l = [2 2 2];
% noise = 0.1;
% col=1-1e-7;
% rng(1)
% X = generate_osman(sz, ranks, ranks_l, noise, 'large_elem', 0);
tol = 1e-14;
no_it =500;
N=length(sz);
[core, wheels, total_time, final_error,er_list,tmlist] = TW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);
cat.TW_ALS.time=tmlist(no_it);
cat.TW_ALS.RE=final_error;

[core, wheels, total_time, final_error,er_list,tmlist] = FTW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);
cat.FTW_ALS.time=tmlist(no_it);
cat.FTW_ALS.RE=final_error;

M=[100 100 3];
[core, wheels, total_time, final_error,er_list,tmlist] = RFTW_ALS(X, ranks, ranks_l,M,'tol',tol, 'maxiters', no_it);
cat.RFTW_ALS.time=tmlist(no_it);
cat.RFTW_ALS.RE=final_error;

mm=2000;
[core, wheels, total_time, nopre_time, final_error, er_list,tmlist]  = TW_ALS_Sampled(X, ranks, ranks_l, mm*ones(size(sz)),4*mm*ones(size(sz)),'tol',tol, 'maxiters', no_it); 
cat.TW_Sampled_ALS.time=tmlist(no_it);
cat.TW_Sampled_ALS.RE=final_error;


%% dog
X=im2double(imread('dog.jpg'));
sz = size(X);
ranks = [4 4 4];
ranks_l = [2 2 2];
% noise = 0.1;
% col=1-1e-7;
% rng(1)
% X = generate_osman(sz, ranks, ranks_l, noise, 'large_elem', 0);
tol = 1e-14;
no_it =500;
N=length(sz);
[core, wheels, total_time, final_error,er_list,tmlist] = TW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);
dog.TW_ALS.time=tmlist(no_it);
dog.TW_ALS.RE=final_error;

[core, wheels, total_time, final_error,er_list,tmlist] = FTW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);
dog.FTW_ALS.time=tmlist(no_it);
dog.FTW_ALS.RE=final_error;

M=[100 100 3];
[core, wheels, total_time, final_error,er_list,tmlist] = RFTW_ALS(X, ranks, ranks_l,M,'tol',tol, 'maxiters', no_it);
dog.RFTW_ALS.time=tmlist(no_it);
dog.RFTW_ALS.RE=final_error;

mm=3000;
[core, wheels, total_time, nopre_time, final_error, er_list,tmlist]  = TW_ALS_Sampled(X, ranks, ranks_l, mm*ones(size(sz)),4*mm*ones(size(sz)),'tol',tol, 'maxiters', no_it); 
dog.TW_Sampled_ALS.time=tmlist(no_it);
dog.TW_Sampled_ALS.RE=final_error;


%% News
load("news.mat");
sz = size(X);
ranks = [3 3 3 3];
ranks_l = [2 2 2 2];
% noise = 0.1;
% col=1-1e-7;
% rng(1)
% X = generate_osman(sz, ranks, ranks_l, noise, 'large_elem', 0);
tol = 1e-14;
no_it =500;
N=length(sz);
[core, wheels, total_time, final_error,er_list,tmlist] = TW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);
news.TW_ALS.time=tmlist(no_it);
news.TW_ALS.RE=final_error;

[core, wheels, total_time, final_error,er_list,tmlist] = FTW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);
news.FTW_ALS.time=tmlist(no_it);
news.FTW_ALS.RE=final_error;

M=[50 50 3 10];
[core, wheels, total_time, final_error,er_list,tmlist] = RFTW_ALS(X, ranks, ranks_l,M,'tol',tol, 'maxiters', no_it);
news.RFTW_ALS.time=tmlist(no_it);
news.RFTW_ALS.RE=final_error;

mm=3000;
[core, wheels, total_time, nopre_time, final_error, er_list,tmlist]  = TW_ALS_Sampled(X, ranks, ranks_l, mm*ones(size(sz)),4*mm*ones(size(sz)),'tol',tol, 'maxiters', no_it); 
news.TW_Sampled_ALS.time=tmlist(no_it);
news.TW_Sampled_ALS.RE=final_error;

%% suzie
load("suzie.mat");
sz = size(X);
ranks = [4 4 4 4];
ranks_l = [2 2 2 2];
% noise = 0.1;
% col=1-1e-7;
% rng(1)
% X = generate_osman(sz, ranks, ranks_l, noise, 'large_elem', 0);
tol = 1e-14;
no_it =500;
N=length(sz);
[core, wheels, total_time, final_error,er_list,tmlist] = TW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);
suzie.TW_ALS.time=tmlist(no_it);
suzie.TW_ALS.RE=final_error;

[core, wheels, total_time, final_error,er_list,tmlist] = FTW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);
suzie.FTW_ALS.time=tmlist(no_it);
suzie.FTW_ALS.RE=final_error;

M=[50 50 3 10];
[core, wheels, total_time, final_error,er_list,tmlist] = RFTW_ALS(X, ranks, ranks_l,M,'tol',tol, 'maxiters', no_it);
suzie.RFTW_ALS.time=tmlist(no_it);
suzie.RFTW_ALS.RE=final_error;

mm=2000;
[core, wheels, total_time, nopre_time, final_error, er_list,tmlist]  = TW_ALS_Sampled(X, ranks, ranks_l, mm*ones(size(sz)),4*mm*ones(size(sz)),'tol',tol, 'maxiters', no_it); 
suzie.TW_Sampled_ALS.time=tmlist(no_it);
suzie.TW_Sampled_ALS.RE=final_error;

%% Indian Pines
load("Indian_pines.mat");
X=indian_pines;
sz = size(X);
ranks = [4 4 4];
ranks_l = [2 2 2];
% noise = 0.1;
% col=1-1e-7;
% rng(1)
% X = generate_osman(sz, ranks, ranks_l, noise, 'large_elem', 0);
tol = 1e-14;
no_it =500;
N=length(sz);
[core, wheels, total_time, final_error,er_list,tmlist] = TW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);
Indian_pines.TW_ALS.time=tmlist(no_it);
Indian_pines.TW_ALS.RE=final_error;

[core, wheels, total_time, final_error,er_list,tmlist] = FTW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);
Indian_pines.FTW_ALS.time=tmlist(no_it);
Indian_pines.FTW_ALS.RE=final_error;

M=[50 50 50];
[core, wheels, total_time, final_error,er_list,tmlist] = RFTW_ALS(X, ranks, ranks_l,M,'tol',tol, 'maxiters', no_it);
Indian_pines.RFTW_ALS.time=tmlist(no_it);
Indian_pines.RFTW_ALS.RE=final_error;

mm=3000;
[core, wheels, total_time, nopre_time, final_error, er_list,tmlist]  = TW_ALS_Sampled(X, ranks, ranks_l, mm*ones(size(sz)),4*mm*ones(size(sz)),'tol',tol, 'maxiters', no_it); 
Indian_pines.TW_Sampled_ALS.time=tmlist(no_it);
Indian_pines.TW_Sampled_ALS.RE=final_error;
