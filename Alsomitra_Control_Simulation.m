function Alsomitra_Control_Simulation(network1,plot_title,nnc)

    % Colin Kessler 4.8.2024 - colinkessler00@gmail.com
    
    
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
    
    constants = load("Normalisation_Constants.mat");
    Cs = constants.constants(1,:);
    Ss = constants.constants(2,:);
    
    % NN or PID controller
    nnc =  false;
    
    ObjectiveFunction = @Alsomitra_nondim;
    
    its = 64;
    epochs = 1;
    obj_f = [];
    
    nn = importNetworkFromONNX(network1,InputDataFormats='BC');
    % nn = 1;
    
    parameters = [5.18218452125279	0.807506506794260	0.105977518471870	4.93681162104530	1.49958010664229	0.238565281050545	2.85289007725274	0.368933365279324	1.73001889433847];
    
    p1 = [-0.3363	0.32178673	13.25541439];
    p2 = [-0.4547	1.117865818	9.362957704];
    p3 = [-0.3311	0.37305935	18.62325016];
    p4 = [-0.4493	0.685756098	7.797849674];
    
    title(plot_title)
    
    data = [];
    
    % Plot simulations over a range of starting y values
    
    for n = 0.1:0.025:0.3
    % for n = -0.3:0.1:0.7
        disp(n)
        hold on
        [data1,errors1,ex_all1] = simulate(n/0.07,parameters,ObjectiveFunction,nn,nnc);
        data = [data;data1];
    end
    % xlim([22 24])
    % ylim([-24 -22])
    
    
    if nnc == false
        Cs = [];
        Ss = [];
        data3 = data;
        data_n = [];
        for i = 1:7
            [N,C,S] = normalize(data(:,i));
            data_n(:,i) = N;
            Cs = [Cs,C];
            Ss = [Ss,S];
        end
        % 
        writematrix(data3,'Training_Data.csv') 
        save('Training_Data','data3')
        % writematrix(data_n,'Training_Data_Normalised.csv') 
        % save('Training_Data_Normalised','data_n')
        % data2 = [Cs; Ss];
        % save('Normalisation_Constants','data2')
    end
    
    
    function [data,errors,ex_all,ds] = simulate(y0,parameters, ObjectiveFunction,nn, nnc)
    
        v_xp0 = 1;
        v_yp0 = 0;
        omega0 = 0;
        theta0 = 0;
        x0 = 0;
        alpha0 = 0;
        error = 0;
        errors = 0;
        errors2 = [];
        error2 = 0;
        nnc = false;
    
        if nnc == true
            % NN CONTROLLER
            % ex = nn.predict(([v_xp0 v_yp0 omega0 theta0 0 0]));
            % input = ([v_xp0 v_yp0 omega0 theta0 x0 y0]- Cs(1:6))./Ss(1:6);
            input = [v_xp0 v_yp0 omega0 theta0 x0 y0];
            ex = nn.predict(input);
            % ex = (ex * Ss(7))+Cs(7);
            % ex = (ex * (0.012)) + 0.181;
            if ex > 0.193
                ex = 0.193;
            elseif ex < 0.181
                ex = 0.181;
            end
        else
            tic
            integral  = sum(errors);
            derivative = errors(end) - error;
            % % PID CONTROLLER
            ex = 0.1850 + (error * 0.1);
            if ex > 0.193
                ex = 0.193;
            elseif ex < 0.181
                ex = 0.181;
            end
            toc
        end
        
        num_sims = 40;
        % num_sims = 80;
        x_scatter = [];
        y_scatter = [];
        
        omega_all = [];
        theta_all = [];
        vx_all =[];
        vy_all =[];
        x_all =[];
        y_all =[];
        ex_all = [];
        alpha_all = [];
        ds = [];
        
        for i = 1:num_sims
    
           
            error = error2;
    
            if nnc == true
                % NN CONTROLLER
                % ex = nn.predict([v_xp0 v_yp0 omega0 theta0 x0 y0 error2] - rot90(Cs) ./ rot90(Ss));
                % input = ([v_xp0 v_yp0 omega0 theta0 x0 y0]- Cs(1:6))./Ss(1:6);
                input = [v_xp0 v_yp0 omega0 theta0 x0 y0];
                ex = nn.predict(input);
                % ex = (ex * Ss(7))+Cs(7);
            else
                % % PID CONTROLLER
                integral  = sum(errors);
                derivative = errors(end) - error;
                
                
                % ex = 0.1870 + (error * 0.008)+ (derivative * 0.02) + (integral * 0.0001);
    
                % if error < 1
                %     error = error * 0.25;
                % end
    
                ex = 0.19 + (error * 0.05)+ (derivative * -0.5)+ (integral * 0.0000);   
                if ex > 0.193
                    ex = 0.193;
                elseif ex < 0.181
                    ex = 0.181;
                end
            end

            if ex > 0.193
                ex = 0.193;
            elseif ex < 0.181
                ex = 0.181;
            end

            
            
            
    
    
            [v_xp, v_yp, omega, theta, x, y, error2] = ObjectiveFunction([parameters,ex],v_xp0, v_yp0, omega0, theta0, x0, y0,num_sims,error2, alpha0);
            
            x_all = [x_all;x];
            y_all = [y_all;y];
        
            v_xp0 = v_xp(end);
            v_yp0 = v_yp(end);
            omega0 = omega(end);
            theta0 = theta(end);
            x0 = x(end);
            y0 = y(end);
            error2 = error2(end);
            % alpha0 = alpha(end);
        
            x_scatter = [x_scatter;x0];
            y_scatter = [y_scatter;y0];
            % error = (y0) - (-1 * (x0));
            
    
            % 
            % BANG-BANG controller
            % if error > 0
            %     ex = 0.106;
            % else
            %     ex = 0.161;
            % end
    
            ex_all = [ex_all;ex];
            theta_all =[theta_all;theta0];
            omega_all =[omega_all;omega0];
            vx_all =[vx_all;v_xp0];
            vy_all =[vy_all;v_yp0];
            x_all =[x_all;x(end)];
            y_all =[y_all;y(end)];
            % alpha_all = [alpha_all;alpha0];
    
           
    
            errors = [errors; error];
            errors2 = [errors2; error2];
       
        
        end
        
        data = [vx_all,vy_all,omega_all,theta_all,x_scatter, y_scatter,ex_all];
        
        % newplot
        % tiledlayout(3,1);
        % nexttile
        % 
        % plot(x_all * 70 / 1000,y_all * 70 / 1000,'red')
        plot(x_all,y_all,'Color',[0.267 0.004 0.329])
        % plot(1:24, abs(omega_all .* (ex_all) * 0.07 ./ vy_all))
    
        hold on
    
        x_c1 = -2:1:30;
        y_c1 = -1 * x_c1;
    
        plot([x_c1] * 1000/70, [y_c1]* 1000/70, '--black')
    
        % % scatter(x_scatter * 70 / 1000,y_scatter * 70 / 1000,4,'blue')
        % % scatter(x_scatter,y_scatter,2,'filled','s')
        % % 
        % xlim([0 25])
        % ylim([-25 5])
        
        xlim([0 45])
        ylim([-50 5])

        % xlim([0 100])
        % ylim([-100 10])

    
        xlabel('x [m]')
        ylabel('y [m]')
        % legend("Simulated trajectories","Desired trajectory")
        daspect([1 1 1])
    
        % nexttile
        % plot(1:61, errors)
        % 
        % nexttile
        % plot(1:61, errors2)
    end

end

function [v_xp, v_yp, omega, theta, x_, y_, error] = Alsomitra_nondim(opt,v_xp0, v_yp0, omega0, theta0, x0, y0,num_sims,error, alpha)
    
    % display(opt)
    
    %% define the initial conditions 
    
    
    % non dimensional period of oscillation T
    T = 2;
    f = 999;
    % number of periods
    n = 20 / num_sims; 
    % n = 40 / num_sims;
    
    % time interval over which to solve the ODEs
    t = 0:0.01:n;
    % fprintf("t end = " + t(end))
    
    % initial conditions for v_xp, v_yp, omega, theta, x, y
    % eps in Matlab gives a value close to  0
    % Y0 = [0; 0; 0; 0; 0.; 0.]; % Cathal's % Y0 = [0.1; 0.1; 0.1; 0.1; 0.1; 0.1]; % mine
    Y0 = [v_xp0, v_yp0, omega0, theta0, x0, y0, 0];

    % Using SI units, instead of the units in LiJFM2022, is not a problem since
    % the code is non dimensional 
    %l = 5.725*10^-2;    % % % p(1) = 5.1*10^-2; % p(1) = 5.1 LiJFM2022 
    %s = 174.5*10^-3;   % Average wingspan
    %m = 0.3175;     % Average mass

    masses = ([0.36,0.37,0.37,0.17])*10^-3;
    lengths = ([54,58,58,59])*10^-3;
    spans = ([162,188,178,170])*10^-3;
    l = mean(lengths);
    l = 7*10^-2;
    s = mean(spans);
    m = mean(masses);

    as= ([64,71,67,68])/2 *10.^-3;
    a = mean(as);
    b = 1/2 *10^-3;

    % h = l*0.002;      % % % p(2) = p(1)*0.002;
    rho_s = 1.4*10^3; % % % p(3) = 1.4*10^3; % p(3) = 1.4 LiJFM2022 
    rho_f = 1.225;    % % % p(4) = 1.225; % p(4) = 1.225*10^-3; LiJFM2022 
    % l_CE = l*0;     % % % p(5) = p(1)*0.0;
    % e_x = 0.08;       % % % p(6) = 0.00;
    % g = 9.80665;      % % % p(7) = 9.80665; % p(7) = 980.665; 
    
  
    % global p 
    p = [l m rho_f a b s]';
    
    [tSol, ySol] = ode45(@(t, y) nondimfreelyfallingplate6(y,opt(10)), t, Y0);
    
    % extract the single variables from the vector with the solutions
    % x component of velocity in the reference system of the body
    v_xp = ySol(:,1);
    
    % y component of velocity in the reference system of the body
    v_yp = ySol(:,2);
    
    % omega, first derivative of theta
    omega = ySol(:,3);
    
    % theta angle defined in Fig.2
    theta = ySol(:,4);
    
    % % x, horizontal coordinate in reference system linked to the lab
    x_ = ySol(:,5);

    % y, horizontal coordinate in reference system linked to the lab
    y_ = ySol(:,6);

    error = ySol(:,7);

    alpha = 1;

end
