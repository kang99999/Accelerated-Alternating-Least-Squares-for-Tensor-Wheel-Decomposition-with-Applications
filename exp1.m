% Experiment 1
clear,clc;
addpath(".\decomposition_code\")
addpath(".\tensor_toolbox-v3.2\")
addpath(".\help_functions\mtimesx\mtimesx_20110223\")

%% N 3 I 30:10:150 R 2 L2 eta 0
for i=3:15
    sz = [10 10 10]*i;
    ranks = [2 2 2];
    ranks_l = [2 2 2];
    noise = 0;
    col=1-1e-7;
    rng(1)
    X = generate_osman(sz, ranks, ranks_l, noise, 'large_elem', 0);

    tol = 1e-14;
    no_it =20;
    N=length(sz);
    [core, wheels, total_time, final_error,er_list,tmlist] = TW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);
    twtm(i-2)=tmlist(20)/20;
    [core, wheels, total_time, final_error,er_list,tmlist] = FTW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);
    ftwtm(i-2)=tmlist(20)/20;
end
fig=figure('Position', [100, 100, 600, 500]);
% 获取默认颜色映射
% colormap_parula = parula(5); % 从 `parula` 颜色映射中获取 3 种颜色
global_default_color_order = get(groot, 'DefaultAxesColorOrder');
% 设置当前图形窗口的颜色顺序
ax = axes(fig);
ax.ColorOrder = global_default_color_order;
hold on;
plot(30:10:150, twtm,'-p','linewidth', 2);
hold on;
plot(30:10:150, ftwtm,'-s','linewidth', 2);
grid on;
xlabel('Dimension size I');
ylabel('Time per iteration [s]');
legend({'TW-ALS','FTW-ALS'},'location','northwest');
set(gca,'fontsize',20);
xlim([30 150]); % 设置插图的x轴范围
hold off


%% N 4 I 10:5:55 R 2 L 2 eta 0

for i=2:11
    sz = [5 5 5 5]*i;
    ranks = [2 2 2 2];
    ranks_l = [2 2 2 2];
    noise = 0;
    col=1-1e-7;
    rng(1)
    X = generate_osman(sz, ranks, ranks_l, noise, 'large_elem', 0);

    tol = 1e-14;
    no_it =20;
    N=length(sz);
    [core, wheels, total_time, final_error,er_list,tmlist] = TW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);
    twtm(i-1)=tmlist(20)/20;
    [core, wheels, total_time, final_error,er_list,tmlist] = FTW_ALS(X, ranks, ranks_l,'tol',tol, 'maxiters', no_it);
    ftwtm(i-1)=tmlist(20)/20;
end
fig=figure('Position', [100, 100, 600, 500]);
% 获取默认颜色映射
% colormap_parula = parula(5); % 从 `parula` 颜色映射中获取 3 种颜色
global_default_color_order = get(groot, 'DefaultAxesColorOrder');
% 设置当前图形窗口的颜色顺序
ax = axes(fig);
ax.ColorOrder = global_default_color_order;
hold on;
plot(10:5:55, twtm(1:10),'-p','linewidth', 2);
hold on;
plot(10:5:55, ftwtm(1:10),'-s','linewidth', 2);
grid on;
xlabel('Dimension size I');
ylabel('Time per iteration [s]');
legend({'TW-ALS','FTW-ALS'},'location','northwest');
set(gca,'fontsize',20);
xlim([10 55]); % 设置插图的x轴范围
hold off

