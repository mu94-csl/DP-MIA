

clc;clear;

close all;





[ 0.5   0.75  1.    1.25  1.5   1.75  2.    2.25  2.5   2.75  3.    3.25  3.5   3.75  4.    4.25  5.   10.   15.   20.   30.   50.  ]

target=[0.7875    0.784375  0.7890625 0.7671875 0.80625   0.746875  0.75  0.746875  0.7296875 0.7203125 0.75625   0.7       0.7234375 0.7 0.703125  0.68125   0.6734375 0.58125   0.5265625 0.5078125 0.546875 0.465625 ]

shadow=[0.7671875 0.7734375 0.7546875 0.7421875 0.7546875 0.759375  0.725 0.7375    0.728125  0.690625  0.725     0.6890625 0.703125  0.6671875 0.65625   0.7015625 0.6453125 0.5859375 0.54375   0.5625    0.5203125 0.528125 ]

attack=[0.56484375 0.5515625  0.55703125 0.54765625 0.5390625  0.53046875 0.54453125 0.5546875  0.54453125 0.55234375 0.5296875  0.55 0.52578125 0.51953125 0.5328125  0.52109375 0.51640625 0.5046875 0.50546875 0.49765625 0.51015625 0.47421875]

eps=[30.549902396000064, 9.412481830217423, 5.026636371461352, 3.4344755876326296, 2.638310473990276, 2.1476073714488724, 1.814977391455928, 1.5751957629098925, 1.3932799016561557, 1.2501097735774191, 1.1342698943580654, 1.0384870726736168, 0.9578902075666982, 0.8888009474584013, 0.8291007564667021, 0.7770393597919716, 0.6542530127874593, 0.3197028794461048, 0.21189716306199452, 0.1668549302436009, 0.10526098716809171, 0.06389947226202096]

skip = 2
















eps = eps(skip:end)

y = target(skip:end)
origAcc = 0.790
maxY = 0.85
ylbl = 'Target Accuracy' 


% y = attack(skip:end)
% origAcc = 0.569
% maxY = 0.58
% ylbl = 'Attack Accuracy' 








 
%% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( eps, y );

% Set up fittype and options.
ft = fittype( 'power2' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
%opts.StartPoint = [0.681099599998994 0.0991118125786401 0.000974152277134997];

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );

% Plot fit with data.
figure( 'Name', 'Regression Curve Fit' );
h = plot( fitresult, xData, yData, 'predobs');
hold on

set(h,'lineWidth',1.75, 'color','#FF5F1F','MarkerEdgeColor','blue');

legend( h, 'Data Points', 'Regression Fit', 'Lower bounds', 'Upper bounds', 'Location', 'southeast', 'Interpreter', 'none' );


g = yline(origAcc ,'-.b','','LineWidth',2, 'DisplayName', 'Acc without DP');

% Label axes
xlabel( 'Epsilon', 'Interpreter', 'none' );
ylabel( ylbl, 'Interpreter', 'none' );

ylim([0.48 maxY])


grid on



