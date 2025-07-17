% Experiment 2
clc,clear
%% N 3 I 100 R2 L2 eta 0

sz = [100 100 100];
ranks = [2 2 2];
ranks_l = [2 2 2];
noise = 0;
col=1-1e-7;
rng(10)
X = generate_osman(sz, ranks, ranks_l, noise, 'large_elem', 0);
tol = 1e-14;
no_it =500;
N=length(sz);

[core, wheels, total_time, final_error,twre,twtm] = TW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);

[core, wheels, total_time, final_error,ftwre,ftwtm] = FTW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);
rng("default")
M=[8 8 8];
[core, wheels, total_time, final_error,rftwre,rftwtm] = RFTW_ALS(X, ranks, ranks_l,M,'tol',tol, 'maxiters', no_it);

figure('Position', [100, 100, 600, 500])
semilogy(twtm(1:2:length(twtm)), twre(1:2:length(twre)),'-p','linewidth', 2);
hold on;
semilogy(ftwtm(1:2:length(ftwtm)), ftwre(1:2:length(ftwre)),'-s','linewidth', 2);
hold on;
semilogy(rftwtm(1:2:length(rftwtm)), rftwre(1:2:length(rftwre)),'-o','linewidth', 2);

hold on;
grid on;

xlabel('Time [s]');
ylabel('RE');
legend({'TW-ALS','FTW-ALS','RFTW-ALS'},'location','northeast');
set(gca,'fontsize',20);
title('Time v.s. RE');

hold off
% 添加插图
axes('Position', [0.6, 0.4, 0.3, 0.25]); % 插图位置和大小
box on; % 为插图添加边框
semilogy(twtm(1:1:length(twtm)), twre(1:1:length(twre)),'-p','linewidth', 2); % 绘制CPRAND曲线
hold on
semilogy(ftwtm(1:5:length(ftwtm)), ftwre(1:5:length(ftwre)),'-s','linewidth', 2); % 绘制CPRAND-Prox曲线
semilogy(rftwtm(1:5:length(rftwtm)), rftwre(1:5:length(rftwtm)),'-o','linewidth', 2);

xlim([0 0.6]); % 设置插图的x轴范围
% ylim([0 1.5]);
set(gca,'fontsize',14);
% ylim([1e-3 1e-2]); % 设置插图的y轴范围
hold off

it_tw=1:length(twre);
it_ftw=1:length(ftwre);
it_rftw=1:length(rftwtm);
figure('Position', [100, 100, 600, 500])
semilogy(it_tw(1:2:length(twre)), twre(1:2:length(twre)),'-p','linewidth', 2);
hold on;
semilogy(it_ftw(1:2:length(ftwre)), ftwre(1:2:length(ftwre)),'-s','linewidth', 2);
hold on;
semilogy(it_rftw(1:2:length(rftwtm)),rftwre(1:2:length(rftwtm)),'-o','linewidth', 2);
hold on;

grid on;
xlabel('Iterations');
ylabel('RE');
legend({'TW-ALS','FTW-ALS','RFTW-ALS'},'location','northeast');
set(gca,'fontsize',20);
title('Iterations v.s. RE');
% title('I=100, R=2');
% xlim([0 50]); % 设置插图的x轴范围
hold off



