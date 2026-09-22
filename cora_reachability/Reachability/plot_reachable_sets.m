% clc; clear all; close all
% load("A:\Big_Downloads\adversarial_model_005_denormonnx_reach_0.mat")
% R1 = R;
% load("A:\Big_Downloads\base_model_denormonnx_reach_0.mat")
% R2 = R;
% load("base_model.onnx_reach_0.2.mat")
% R2 = R;

% Plot preamble.
set(0, 'defaultFigureRenderer', 'painters')
set(0,'DefaultFigureWindowStyle','docked')
font=12;
set(groot, 'defaultAxesTickLabelInterpreter', 'latex'); 
set(groot, 'defaultLegendInterpreter', 'latex');
set(0,'defaultTextInterpreter','latex');
set(0, 'defaultAxesFontSize', font)
set(0, 'defaultLegendFontSize', font)
set(0, 'defaultAxesFontName', 'Times New Roman');
set(0, 'defaultLegendFontName', 'Times New Roman');
set(0, 'DefaultLineLineWidth', 0.5);

dims = [5 6];
f1 = figure ;
tiledlayout(1,3); nexttile


xlabel('$x$ [m]'); ylabel('$y$ [m]');
axis square; grid off
xlim([0 45])
ylim([-45 5])

hold on; box on;
projDim = dims;
h1 = [];

tic
h1_1 =  plot(R2,projDim,'DisplayName','Reachable set','Unify',true,'UnifyTotalSets',1,'FaceColor', [0.6928    0.1651    0.5645],'EdgeColor', 'k','FaceAlpha',0.6,'EdgeAlpha',1);
toc

tic
h2_1 =  plot(R1,projDim,'DisplayName','Reachable set','Unify',true,'UnifyTotalSets',1,'FaceColor', [0.9883    0.6523    0.2114],'EdgeColor', 'k','FaceAlpha',0.5,'EdgeAlpha',1);
toc

x_c1 = -2:1:30;
y_c1 = -1 * x_c1;
h3_1 = plot([x_c1] * 1000/70, [y_c1]* 1000/70, '--black');

h4_1 = plot(polyshape([0 0 50 50], [-2 2 -48 -52]),'FaceColor', [0.800, 1.000, 0.800],'EdgeColor', 'k',"LineStyle", ":",'FaceAlpha',0,'EdgeAlpha',1);

lgd = legend([h3_1 h4_1 h1_1 h2_1], {"Target Trajectory","Target Region","Naive NN","Adversarial NN"});
lgd.Layout.Tile = 3;

nexttile
hold on; box on;useCORAcolors('CORA:contDynamics');

xlabel('$x$ [m]'); 
axis square; grid off
xlim([35 45])
ylim([-45 -35])

tic
h1 = plot(R2,projDim,'DisplayName','Reachable set','Unify',true,'UnifyTotalSets',1,'FaceColor', [0.6928    0.1651    0.5645],'EdgeColor', 'k','FaceAlpha',0.6,'EdgeAlpha',1);
toc

tic
h2 = plot(R1,projDim,'DisplayName','Reachable set','Unify',true,'UnifyTotalSets',1,'FaceColor', [0.9883    0.6523    0.2114],'EdgeColor', 'k','FaceAlpha',0.5,'EdgeAlpha',1);
toc

x_c1 = -2:1:30;
y_c1 = -1 * x_c1;
h3 = plot((x_c1) * 1000/70, (y_c1)* 1000/70, '--black');

h4 = plot(polyshape([0 0 50 50], [-2 2 -48 -52]),'FaceColor', [0.800, 1.000, 0.800],'EdgeColor', 'k',"LineStyle", ":",'FaceAlpha',0,'EdgeAlpha',1);

% 
% nexttile
% axis square; grid off
% 
% tic
% h1 = plotOverTime(R2,4,'DisplayName','Reachable set','Unify',true,'UnifyTotalSets',1,'FaceColor', [0.6928    0.1651    0.5645],'EdgeColor', 'k','FaceAlpha',0.6,'EdgeAlpha',1);
% toc
% 
% tic
% h2 = plotOverTime(R1,4,'DisplayName','Reachable set','Unify',true,'UnifyTotalSets',1,'FaceColor', [0.9883    0.6523    0.2114],'EdgeColor', 'k','FaceAlpha',0.5,'EdgeAlpha',1);
% toc


exportgraphics(f1,'reachability_2.png','Resolution',600)