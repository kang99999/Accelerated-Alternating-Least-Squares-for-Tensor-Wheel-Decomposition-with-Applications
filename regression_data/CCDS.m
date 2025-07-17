%CCDS
clc
clear


load('climateP17.mat')
sigma = 0.1; 
[nLoc, tLen] = size(series{1});

global verbose
verbose = 1;
global evaluate
evaluate = 3;

nLag = 3;
nTask = length(series);
A = zeros(nLoc, nLoc*nLag, nTask);
X = cell(nTask, 1);
Y = cell(nTask, 1);
test.X = cell(nTask, 1);
test.Y = cell(nTask, 1);

mu = logspace(-1, 1.3, 10);

for i = 1:nTask
    Y{i} = series{i}(:, nLag+1:tLen);
    X{i} = zeros(nLag*nLoc, (tLen - nLag));
    for ll = 1:nLag
        X{i}(nLoc*(ll-1)+1:nLoc*ll, :) = series{i}(:, nLag+1-ll:tLen-ll);
    end
end



xdata=zeros(17,125,3,153);
ydata=zeros(17,125,153);
for i = 1:size(X,1)
    xdata(i,:,:,:)=reshape(X{i},[125,3,153]);
    ydata(i,:,:,:)=reshape(Y{i},[125,153]);
end
xdata=permute(xdata,[4,2,1,3]);
ydata=permute(ydata,[3,2,1]);

k = 153;
xdata = xdata(1:k,:,:,:);
ydata = ydata(1:k,:,:);
% 样本数
train_numm = round(k/10*9);
test_num=k-train_numm;
%train_num = round(train_numm/5*4);
%val_num=train_numm-train_num;
train_num = train_numm;
val_num = 0;
total_samples_num = train_num+val_num+test_num;

    
%随机顺序
% random_indices = randperm(total_samples_num);
x=xdata(1:total_samples_num,:,:,:);
y=ydata(1:total_samples_num,:,:);
% x=x(random_indices,:,:,:);
% y=y(random_indices,:,:);

%划分数据集
trainx=x(1:train_num,1:20,:,:);
trainy=y(1:train_num,1:20,:,:);
valx=x(train_num+1:train_num+val_num,1:20,:,:);
valy=y(train_num+1:train_num+val_num,1:20,:,:);
testx=x(train_num+val_num+1:total_samples_num,1:20,:,:);
testy=y(train_num+val_num+1:total_samples_num,1:20,:,:);

%存储数据
% save(['U17_trainx','.mat'], 'trainx');
% save(['U17_trainy','.mat'], 'trainy');
% save(['U17_testx','.mat'], 'testx');
% save(['U17_testy','.mat'], 'testy');
% save(['U17_valx','.mat'], 'valx');
% save(['U17_valy','.mat'], 'valy');