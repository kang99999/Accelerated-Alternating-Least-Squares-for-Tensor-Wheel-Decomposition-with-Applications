% Experiment 3 or 4
clc,clear
%% taking salinas as an example
load("Salinas.mat")
X=salinas;
sz = size(X);
ranks = [4 4 4];
ranks_l = [2 2 2];
% noise = 0.1;
% col=1-1e-7;
% rng(1)
% X = generate_osman(sz, ranks, ranks_l, noise, 'large_elem', 0);
tol = 1e-14;
no_it =100;
N=length(sz);
J=20;
%% FTW-ALS
for j=1:J
    for i=1:1

    [core, wheels, total_time, final_error,er_list,tmlist] = FTW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);

    
    ftwtm(j,i)=tmlist(end);
    ftwre(j,i)=final_error;
    end
end
mean_ftwre=mean(ftwre,1);
mean_ftwtm=mean(ftwtm,1);

%% RFTW-ALS
for j=1:J
    for i=1:8
        M=[10 10 10]*i+20;
        
        [core, wheels, total_time, final_error,er_list,tmlist] = RFTW_ALS(X, ranks, ranks_l,M,'tol',tol, 'maxiters', no_it);

        Rftwtm(j,i)=tmlist(end);
        Rftwre(j,i)=final_error;
    end
end
mean_Rftwre=mean(Rftwre,1);
mean_Rftwtm=mean(Rftwtm,1);

%% TW-Sampled-ALS
for j=1:J
    for i=1:12

         mm=1000*i;

        [core, wheels, total_time, nopre_time, final_error, er_list,tmlist]  = TW_ALS_Sampled(X, ranks, ranks_l, mm*ones(size(sz)),4*mm*ones(size(sz)),'tol',tol, 'maxiters', no_it); 
        Stwtm(j,i)=tmlist(end);
        Stwre(j,i)=final_error;
    end
end
mean_Stwre=mean(Stwre,1);
mean_Stwtm=mean(Stwtm,1);

%% plotting
%%
figure('Position', [100, 100, 600, 500])
% 指定每组数据的标签
labels = {  '30','40','50','60','70','80','90','100'};

boxplot(Rftwre, 'Labels', labels)
% 添加标题和轴标签
title('Sketch size v.s. RE');
xlabel('Sketch size M',"FontSize",20);
ylabel('RE','FontSize',20);
set(gca, 'FontSize', 20);
% 显示图形
grid on; % 添加网格线
hold on

semilogy(1:8,mean_Rftwre,'k-.','LineWidth', 1.5,'DisplayName', 'Mean RE (RFTW-ALS)')
semilogy(1:8,mean_ftwre*ones(1,8),'b-s','LineWidth', 1.5,'DisplayName', 'Mean RE (FTW-ALS)')
legend('Location', 'northeast');
hold off
%%
figure('Position', [100, 100, 600, 500])
% 指定每组数据的标签

labels = {   '30','40','50','60','70','80','90','100'};
boxplot(Rftwtm, 'Labels', labels)
% 添加标题和轴标签
title('Sketch size v.s. Time');
% title('100 \times100 \times 100');
xlabel('Sketch size M',"FontSize",20);
ylabel('Time [s]','FontSize',20);
set(gca, 'FontSize', 20);
% 显示图形
grid on; % 添加网格线
hold on

plot(1:8,mean_Rftwtm,'k-.','LineWidth', 1.5,'DisplayName', 'Mean Time (RFTW-ALS)')
plot(1:8,mean_ftwtm*ones(1,8),'b-s','LineWidth', 1.5,'DisplayName', 'Mean Time (FTW-ALS)')
legend('Location', 'northwest');
ylim([0 50])
hold off
%%
figure('Position', [100, 100, 600, 500])
% 指定每组数据的标签
labels = {  '1000','2000','3000','4000','5000','6000','7000','8000','9000','10000','11000','12000'};

boxplot(Stwre, 'Labels', labels)
% 添加标题和轴标签
title('Sketch size v.s. RE');
xlabel('Sketch size m1',"FontSize",20);
ylabel('RE','FontSize',20);
set(gca, 'FontSize', 20);
% 显示图形
grid on; % 添加网格线
hold on

semilogy(1:12,mean_Stwre,'k-.','LineWidth', 1.5,'DisplayName', 'Mean RE (TW-Sampled-ALS)')
hold on
semilogy(1:12,mean_ftwre*ones(1,12),'b-s','LineWidth', 1.5,'DisplayName', 'Mean RE (FTW-ALS)')
semilogy(1:12,mean_Rftwre(8)*ones(1,12),'r-o','LineWidth', 1.5,'DisplayName', 'Mean RE (RFTW-ALS (M=100))')

legend('Location', 'northeast');
hold off

%%
figure('Position', [100, 100, 600, 500])
% 指定每组数据的标签
% labels = {  '10', '15','20','25','30','35','40','45','500', '750','1000','1250','1500','1750','2000','2250','FTW-ALS'};
labels = {  '1000', '2000','3000','4000','5000','6000','7000','8000','9000','10000','11000','12000'};
boxplot(Stwtm, 'Labels', labels)
% 添加标题和轴标签
title('Sketch size v.s. Time');

xlabel('Sketch size m1',"FontSize",20);
ylabel('Time [s]','FontSize',20);
set(gca, 'FontSize', 20);
% 显示图形
grid on; % 添加网格线
hold on

plot(1:12,mean_Stwtm,'k-.','LineWidth', 1.5,'DisplayName', 'Mean Time (TW-Sampled-ALS)')
plot(1:12,mean_ftwtm*ones(1,12),'b-s','LineWidth', 1.5,'DisplayName', 'Mean Time (FTW-ALS)')
plot(1:12,mean_Rftwtm(8)*ones(1,12),'r-o','LineWidth', 1.5,'DisplayName', 'Mean Time (RFTW-ALS (M=100))')
legend('Location', 'northwest');
hold off


