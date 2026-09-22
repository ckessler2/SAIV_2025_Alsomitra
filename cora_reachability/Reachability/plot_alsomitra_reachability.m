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

goalSet1 = interval([-999;-999;-999;-999;0;0;-999;],[999;999;999;999;0;0;999;]);
c = [0;0;0;0;0;0;0];
G = [0 0;0 0;0 0;0 0;0.3 100;0.3 -100;0 0];
goalSet2 = zonotope(c,G);
goalSet = plus(goalSet1,goalSet2);

goalSet3 = interval([-999;-999;-999;-999;0;0;-sqrt(2)*0.3 * 0.07;],[999;999;999;999;0;0;sqrt(2)*0.3 * 0.07;]);

spec = specification(goalSet, 'safeSet',interval(0, params.tFinal));

dims = [5 6];
f1 = figure ;
tiledlayout(1,3); nexttile

hold on; box on;
projDim = dims;
h1 = [];

h0 = plot(spec, dims,'DisplayName', 'Goal set');
% R = X;

for i = 1:length(R)
    h1 =  plot(R(i),projDim,'DisplayName','Reachable set','Unify',true,'UnifyTotalSets',5,'FaceColor', [0 0.4470 0.7410]);
end

% plot initial set
h2 = plot(R(1).R0,projDim, 'DisplayName','Initial set','FaceColor', [1 1 1]);

% plot simulation results      
h3 = plot(simRes1,projDim,'DisplayName','Simulations','color','k');

x_c1 = -2:1:30;
y_c1 = -1 * x_c1;

h4 = plot([x_c1] * 1000/70, [y_c1]* 1000/70, '--black')

% label plot

lgd = legend([h0 h1 h2 h3 h4], {"Goal Set", "Reachable set",'Initial set', 'Simulations', 'Desired Trajectory'});
lgd.Layout.Tile = 3;

grid on
daspect([1 1 1])

% labels and legend
xlabel('$x$ (m)'); ylabel('$y$ (m)');
title("$x$ vs $y$")


% scatter(x_scatter * 70 / 1000,y_scatter * 70 / 1000,4,'blue')
% scatter(x_scatter,y_scatter,2,'filled','s')
axis square;
xlim([0 40])
ylim([-40 5])

nexttile
hold on; box on;useCORAcolors('CORA:contDynamics');

% spec = specification(goalSet3, 'safeSet',interval(0, params.tFinal));
% plotOverTime(spec, 7, 'DisplayName', 'Goal set');
% 
% plotOverTime(R, 7, 'DisplayName', 'Reachable set','FaceColor', [0 0.4470 0.7410]);
% 
% plotOverTime(simRes1, 7, 'DisplayName', 'Simulations','color','k');
% 
% plot([0 params.tFinal],[0 0],'--black')
% xlabel('$t$(s)');
% ylabel(['$y$ error (m)']);
% title("$y$ error vs time")
% axis square;
% xlim([0 params.tFinal])
% grid on
% 
% exportgraphics(f1,'Reach1.png','Resolution',600)