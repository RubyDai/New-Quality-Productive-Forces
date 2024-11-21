%% Clear environment variables
warning off             % Close alarm message
close all               % Close open figures
clear                   % Clear variables
clc                     % Clear the command line

%% Data input: ID, Variable and years
data = readtable('Test_Data.xlsx');  % Importing data containing both text and numbers using readtable

%% Get all IDs
idNames = unique(data.ID);

%% Initialize a table to store all prediction results
allPredictions = table();

%% Rolling window estimation
% This process uses a rolling window of size `kim` to construct input features,
% and predicts the value at a `zim` step ahead. The window length determines
% how much historical data is utilized for each prediction.
%% Loop through each ID
for i = 1:length(idNames)
    id = idNames{i};
    % Extract data for the current ID
    idData = data(strcmp(data.ID, id), :);

    %% Extract variables and check data length
    result = idData.Variable;
    num_samples = length(result);  % Number of samples 
    kim = 5;                       % Delay step length (kim historical data as independent variables)
    zim = 1;                       % Forecast across zim time points

    if num_samples < kim + zim
        fprintf('Insufficient data for ID %s, skipping.\n', id);
        continue;  % Insufficient data, skipping this ID
    end

    %% Constructing the dataset
    res = [];
    for j = 1:num_samples - kim - zim + 1
        res(j, :) = [reshape(result(j:j + kim - 1), 1, kim), result(j + kim + zim - 1)];
    end

    %% Dataset analysis
    outdim = 1;                                    % The last column is the output
    num_size = 0.7;                                % The proportion of training set to data set
    num_samples_res = size(res, 1);                % The number of samples after construction
    num_train_s = round(num_size * num_samples_res); % Number of training set samples
    f_ = size(res, 2) - outdim;                    % Input feature dimensions

    %% Divide the training set and test set
    P_train = res(1:num_train_s, 1:f_)';
    T_train = res(1:num_train_s, f_ + 1:end)';

    P_test = res(num_train_s + 1:end, 1:f_)';
    T_test = res(num_train_s + 1:end, f_ + 1:end)';

    %% Data Normalization
    [P_train_norm, ps_input] = mapminmax(P_train, 0, 1);
    P_test_norm = mapminmax('apply', P_test, ps_input);

    [t_train_norm, ps_output] = mapminmax(T_train, 0, 1);
    t_test_norm = mapminmax('apply', T_test, ps_output);

    %% Convert the data format to suitable input for LSTM
    M = size(P_train_norm, 2);
    N = size(P_test_norm, 2);

    % Convert the input data into a cell array, where each element is a time series
    p_train = cell(M, 1);
    for k = 1:M
        p_train{k} = P_train_norm(:, k);
    end

    p_test = cell(N, 1);
    for k = 1:N
        p_test{k} = P_test_norm(:, k);
    end

    t_train_norm = t_train_norm';
    t_test_norm = t_test_norm';

    %% Creating an LSTM model
    layers = [
        sequenceInputLayer(f_)              % Input layer, feature dimension is f_
        lstmLayer(10, 'OutputMode', 'last') % LSTM layer, outputs the result of the last time step
        reluLayer                           % ReLU activation layer
        fullyConnectedLayer(1)              % Fully connected layer, output dimension is 1
        regressionLayer];                   % Regression Layer

    %% Setting training options
    options = trainingOptions('adam', ...                 % Optimization algorithm Adam
        'MaxEpochs', 300, ...                             % Maximum number of training sessions
        'GradientThreshold', 1, ...                       % Gradient Threshold
        'InitialLearnRate', 5e-3, ...                     % Initial learning rate
        'LearnRateSchedule', 'piecewise', ...             % Learning rate adjustment strategy
        'LearnRateDropPeriod', 250, ...                   % Learning rate reduction cycle
        'LearnRateDropFactor', 0.1, ...                   % Learning rate reduction factor
        'L2Regularization', 1e-4, ...                     % Regularization parameter
        'ExecutionEnvironment', 'auto', ...               % Training Environment
        'Verbose', false);                                % Do not display detailed training information

    %% Training the model
    net = trainNetwork(p_train, t_train_norm, layers, options);

    %% Simulation prediction
    t_sim_train = predict(net, p_train);
    t_sim_test = predict(net, p_test);

    %% Data denormalization
    T_sim_train = mapminmax('reverse', t_sim_train', ps_output);
    T_sim_test = mapminmax('reverse', t_sim_test', ps_output);

    %% Preparing prediction results
    % Calculate the index of the training set and test set
    train_indices = (1:num_train_s) + kim + zim - 1;
    test_indices = (num_train_s + 1:num_samples_res) + kim + zim - 1;

    % Get the corresponding year
    years_train = idData.years(train_indices);
    years_test = idData.years(test_indices);

    %% Make sure the variable is a column vector
    T_train = T_train(:);
    T_sim_train = T_sim_train(:);
    years_train = years_train(:);

    T_test = T_test(:);
    T_sim_test = T_sim_test(:);
    years_test = years_test(:);

    %% Create a table of prediction results for the current ID
    num_train = length(T_sim_train);
    num_test = length(T_sim_test);

    id_column_train = repmat({id}, num_train, 1);
    id_column_test = repmat({id}, num_test, 1);

    id_predictions_train = table(id_column_train, years_train, T_train, T_sim_train, ...
        'VariableNames', {'ID', 'Year', 'Actual', 'Predicted'});

    id_predictions_test = table(id_column_test, years_test, T_test, T_sim_test, ...
        'VariableNames', {'ID', 'Year', 'Actual', 'Predicted'});

    % Combine the prediction results of the training set and the test set
    id_predictions = [id_predictions_train; id_predictions_test];

    %% Combine all predictions
    allPredictions = [allPredictions; id_predictions];
end

%% Display or save all prediction results
disp(allPredictions);
writetable(allPredictions, 'Output.xlsx');