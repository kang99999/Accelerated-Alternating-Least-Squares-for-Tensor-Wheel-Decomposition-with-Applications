clear,clc;
addpath(".\regression_code\")
addpath(".\regression_data\")

%% CCDS
para.P = [20 17 3];
para.Q = [20,17];
para.L = length(para.P);
para.M = length(para.Q);
para.dim = [para.P,para.Q];
para.N = length(para.dim);
para.maxiter=150;
para.datarep=20;
para.lambda = 1e3;


load('CCDS_trainx.mat', 'trainx');
load('CCDS_trainy.mat', 'trainy');
load('CCDS_testx.mat', 'testx');
load('CCDS_testy.mat', 'testy');
load('CCDS_valx.mat', 'valx');
load('CCDS_valy.mat', 'valy');
rank = [2 2 2 2 2];
rank_l=[2 2 2 2 2];

CCDS_Result=[];

% TWRR-ALS
[est_model,runtime_ttr,tmlist] = TWRR_ALS(para, rank, rank_l, trainx, trainy, valx, valy);
est_testy=contract(testx,est_model,para.L);
Ypred=reshape(est_testy,[numel(testy),1]);
Y=reshape(testy,[numel(testy),1]);

cor = mycorrcoef(Ypred(:),Y(:));
Ypress = sum((Y(:)-Ypred(:)).^2);
rmse  = sqrt(Ypress./numel(Y));
Q2 = 1 - Ypress./sum(Y(:).^2);

result=struct('algorithm','TWRR_ALS','cor',cor,'q2',Q2,'rmse',rmse,'time',tmlist(length(tmlist)));
CCDS_Result=[CCDS_Result result];

% FTWRR-ALS
[est_model,runtime_ttr,tmlist] = FTWRR_ALS(para, rank, rank_l, trainx, trainy, valx, valy);
est_testy=contract(testx,est_model,para.L);
Ypred=reshape(est_testy,[numel(testy),1]);
Y=reshape(testy,[numel(testy),1]);

cor = mycorrcoef(Ypred(:),Y(:));
Ypress = sum((Y(:)-Ypred(:)).^2);
rmse  = sqrt(Ypress./numel(Y));
Q2 = 1 - Ypress./sum(Y(:).^2);

result=struct('algorithm','FTWRR_ALS','cor',cor,'q2',Q2,'rmse',rmse,'time',tmlist(length(tmlist)));
CCDS_Result=[CCDS_Result result];

% RFTWRR-ALS
M=[15 10 3 15 10];
 [est_model,runtime,tmlist] = RFTWRR_ALS(para, rank, rank_l,M, trainx, trainy, valx, valy);

est_testy=contract(testx,est_model,para.L);
Ypred=reshape(est_testy,[numel(testy),1]);
Y=reshape(testy,[numel(testy),1]);

cor = mycorrcoef(Ypred(:),Y(:));
Ypress = sum((Y(:)-Ypred(:)).^2);
rmse  = sqrt(Ypress./numel(Y));
Q2 = 1 - Ypress./sum(Y(:).^2);

result=struct('algorithm','RFTWRR_ALS','cor',cor,'q2',Q2,'rmse',rmse,'time',tmlist(length(tmlist)));
CCDS_Result=[CCDS_Result result];

%% USHCN
para.P = [108 3 3];
para.Q = [108,3];
para.L = length(para.P);
para.M = length(para.Q);
para.dim = [para.P,para.Q];
para.N = length(para.dim);
para.maxiter=150;
para.datarep=20;
para.lambda = 1e4;


load('USHCN_trainx.mat', 'trainx');
load('USHCN_trainy.mat', 'trainy');
load('USHCN_testx.mat', 'testx');
load('USHCN_testy.mat', 'testy');
load('USHCN_valx.mat', 'valx');
load('USHCN_valy.mat', 'valy');
rank = [2 2 2 2 2];
rank_l=[2 2 2 2 2];

USHCN_Result=[];

% % TWRR-ALS
% [est_model,runtime_ttr,tmlist] = TWRR_ALS(para, rank, rank_l, trainx, trainy, valx, valy);
% est_testy=contract(testx,est_model,para.L);
% Ypred=reshape(est_testy,[numel(testy),1]);
% Y=reshape(testy,[numel(testy),1]);
% 
% cor = mycorrcoef(Ypred(:),Y(:));
% Ypress = sum((Y(:)-Ypred(:)).^2);
% rmse  = sqrt(Ypress./numel(Y));
% Q2 = 1 - Ypress./sum(Y(:).^2);
% 
% result=struct('algorithm','TWRR_ALS','cor',cor,'q2',Q2,'rmse',rmse,'time',tmlist(length(tmlist)));
% CCDS_Result=[CCDS_Result result];

% FTWRR-ALS
[est_model,runtime_ttr,tmlist] = FTWRR_ALS(para, rank, rank_l, trainx, trainy, valx, valy);
est_testy=contract(testx,est_model,para.L);
Ypred=reshape(est_testy,[numel(testy),1]);
Y=reshape(testy,[numel(testy),1]);

cor = mycorrcoef(Ypred(:),Y(:));
Ypress = sum((Y(:)-Ypred(:)).^2);
rmse  = sqrt(Ypress./numel(Y));
Q2 = 1 - Ypress./sum(Y(:).^2);

result=struct('algorithm','FTWRR_ALS','cor',cor,'q2',Q2,'rmse',rmse,'time',tmlist(length(tmlist)));
USHCN_Result=[USHCN_Result result];

% RFTWRR-ALS
M=[50 3 3 50 3];
 [est_model,runtime,tmlist] = RFTWRR_ALS(para, rank, rank_l,M, trainx, trainy, valx, valy);

est_testy=contract(testx,est_model,para.L);
Ypred=reshape(est_testy,[numel(testy),1]);
Y=reshape(testy,[numel(testy),1]);

cor = mycorrcoef(Ypred(:),Y(:));
Ypress = sum((Y(:)-Ypred(:)).^2);
rmse  = sqrt(Ypress./numel(Y));
Q2 = 1 - Ypress./sum(Y(:).^2);

result=struct('algorithm','RFTWRR_ALS','cor',cor,'q2',Q2,'rmse',rmse,'time',tmlist(length(tmlist)));
USHCN_Result=[USHCN_Result result];


%% ECoG
para.P = [64 10 10];
para.Q = [3,6];
para.L = length(para.P);
para.M = length(para.Q);
para.dim = [para.P,para.Q];
para.N = length(para.dim);
para.maxiter=150;
para.datarep=20;
para.lambda = 1e2;

load('ECoG_trainx.mat', 'trainx');
load('ECoG_trainy.mat', 'trainy');
load('ECoG_testx.mat', 'testx');
load('ECoG_testy.mat', 'testy');
load('ECoG_valx.mat', 'valx');
load('ECoG_valy.mat', 'valy');
trainx=permute(trainx,[4,1,2,3]);
testx=permute(testx,[4,1,2,3]);
valx=permute(valx,[4,1,2,3]);
trainy=(trainy-(-9.9161))/(199.7508);
testy=(testy-(-9.9161))/(199.7508);
valy=(valy-(-9.9161))/(199.7508);

rank = [2 2 2 2 2];
rank_l=[2 2 2 2 2];

ECoG_Result=[];

% TWRR-ALS
[est_model,runtime_ttr,tmlist] = TWRR_ALS(para, rank, rank_l, trainx, trainy, valx, valy);
est_testy=contract(testx,est_model,para.L);
Ypred=reshape(est_testy,[numel(testy),1]);
Y=reshape(testy,[numel(testy),1]);

cor = mycorrcoef(Ypred(:),Y(:));
Ypress = sum((Y(:)-Ypred(:)).^2);
rmse  = sqrt(Ypress./numel(Y));
Q2 = 1 - Ypress./sum(Y(:).^2);

result=struct('algorithm','TWRR_ALS','cor',cor,'q2',Q2,'rmse',rmse,'time',tmlist(length(tmlist)));
ECoG_Result=[ECoG_Result result];

% FTWRR-ALS
[est_model,runtime_ttr,tmlist] = FTWRR_ALS(para, rank, rank_l, trainx, trainy, valx, valy);
est_testy=contract(testx,est_model,para.L);
Ypred=reshape(est_testy,[numel(testy),1]);
Y=reshape(testy,[numel(testy),1]);

cor = mycorrcoef(Ypred(:),Y(:));
Ypress = sum((Y(:)-Ypred(:)).^2);
rmse  = sqrt(Ypress./numel(Y));
Q2 = 1 - Ypress./sum(Y(:).^2);

result=struct('algorithm','FTWRR_ALS','cor',cor,'q2',Q2,'rmse',rmse,'time',tmlist(length(tmlist)));
ECoG_Result=[ECoG_Result result];

% RFTWRR-ALS
M=[30 10 10 3 6];
 [est_model,runtime,tmlist] = RFTWRR_ALS(para, rank, rank_l,M, trainx, trainy, valx, valy);

est_testy=contract(testx,est_model,para.L);
Ypred=reshape(est_testy,[numel(testy),1]);
Y=reshape(testy,[numel(testy),1]);

cor = mycorrcoef(Ypred(:),Y(:));
Ypress = sum((Y(:)-Ypred(:)).^2);
rmse  = sqrt(Ypress./numel(Y));
Q2 = 1 - Ypress./sum(Y(:).^2);

result=struct('algorithm','RFTWRR_ALS','cor',cor,'q2',Q2,'rmse',rmse,'time',tmlist(length(tmlist)));
ECoG_Result=[ECoG_Result result];
