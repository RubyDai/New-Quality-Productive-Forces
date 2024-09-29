%%  Clear environment variables
warning off             % Turn off alarm message
close all               % Close opened figures
clear                   % Clear variables
clc                     % Clear the command line

%%Construct an empty table and put the results into this table
NQPDATA = table();

% Reading Files (Excel format is strongly recommended)
% The data file we use here is an excel file with three column: YEAR,
% REGION, and GTFPEfficiency
data = readtable('Your file path');
data = sortrows(data, "YEAR" , "ascend" ); % Rank by YEAR
data.REGION = string(data.REGION);

% Get a unique list of regions, as each region has a different growth path,
% requiring us to refit the model for each region.
regions = unique(data.REGION);


% Loop through each region
for i = 1:length(regions)
    region = regions(i);
    
    % Get data for the current region
    regionData = data(strcmp(data.REGION, region), :);
    
    % Extracting time series data
    result = regionData.GTFPEfficiency;

%%  Data analysis
num_samples = length(result);  % Number of samples
kim = 5;                       % Delay step length (using kim historical data as independent variables)
zim =  1;                      % Predict across zim time period

%%  Divide the dataset
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(result(i: i + kim - 1), 1, kim), result(i + kim + zim - 1)];
end

%%  Dataset analysis
outdim = 1;                                  % The last column is the output
num_size = 0.3;                              % The proportion of training set to data set
                                             % We set 0.3 here for we would
                                             % like more test set
num_train_s = round(num_size * num_samples); % Number of training set
f_ = size(res, 2) - outdim;                  % Input feature dimensions

%%  Divide the training set and test set
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%%  Data Normalization
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%% Data Flattening
% Flattening the data into 1D is just one processing method.
% It can also be flattened into 2D or 3D data, requiring modifications to the corresponding model structure.
% However, it should always remain consistent with the input layer's data structure.
P_train =  double(reshape(P_train, f_, 1, 1, M));
P_test  =  double(reshape(P_test , f_, 1, 1, N));

t_train = t_train';
t_test  = t_test' ;

%%  Data format conversion
for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end

%%  Creating the Model
layers = [
    sequenceInputLayer(f_)              % Building the input layer
    
    lstmLayer(10, 'OutputMode', 'last') % LSTM layer
    reluLayer                           % Relu layer
    
    fullyConnectedLayer(1)              % Fully connected layer
    regressionLayer];                   % Regression layer

%%  Parameter settings
options = trainingOptions('adam', ...                 % Optimization algorithm Adam
    'MaxEpochs', 300, ...                             % Maximum times of training 
    'GradientThreshold', 1, ...                       % Gradient Threshold
    'InitialLearnRate', 5e-3, ...                     % Initial learning rate
    'LearnRateSchedule', 'piecewise', ...             % Learning rate adjustment
    'LearnRateDropPeriod', 250, ...                   % After training 250 times, start adjusting the learning rate
    'LearnRateDropFactor',0.1, ...                    % Learning rate adjustment factor
    'L2Regularization', 1e-4, ...                     % Regularization parameter
    'ExecutionEnvironment', 'auto',...                % Training Environment
    'Verbose', false );                               % Turn off optimization process

%%  Training the model
net = trainNetwork(p_train, t_train, layers, options);

%%  Simulation prediction
t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test );

%%  Data denormalization: T_sim1 is the training set, T_sim2 is the test set
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  Root mean square error
T_sim1 = array2table(T_sim1);
T_sim2 = array2table(T_sim2);
T_sim1.Properties.VariableNames = {'NQPF'};
T_sim2.Properties.VariableNames = {'NQPF'};
NQPDATA = [NQPDATA; T_sim1];
NQPDATA = [NQPDATA; T_sim2];

end


% Display results
disp(NQPDATA);
filename = 'Your file name';  % Define the file name
writetable(NQPDATA, filename );  % Writing to File
writetable(data, 'Your another file name' );