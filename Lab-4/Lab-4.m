%% Lab 4: Handwritten Digit Recognition using Neural Networks

% Clear workspace
clear; close all; clc;

%% Step 1: Load and Preprocess the Image
imageFile = 'numberImage.png';
numRows = 5;
try
    trainingFeatures = pozymiai_raidems_atpazinti(imageFile, numRows);
catch exception
    fprintf('Error during feature extraction: %s\n', exception.message);
    return;
end

%% Step 2: Prepare Training Data
% Convert features from cell array to matrix
P = cell2mat(trainingFeatures);

% Create target matrix for training (one-hot encoding for digits 0-9)
numDigits = 10; % Total digits (0-9)
numSamplesPerDigit = size(P, 2) / numDigits; % Calculate samples per digit
if mod(size(P, 2), numDigits) ~= 0
    error('Mismatch in the number of samples and expected digits. Check your data.');
end
T = repmat(eye(numDigits), 1, numSamplesPerDigit);

%% Step 3: Train the RBF Neural Network
% Reduce the number of neurons from 13 to a smaller number
numNeurons = 5; % Adjust this value as needed
net = newrb(P, T, 0, 1, numNeurons);

%% Step 4: Test the Network
% Use the same image for testing or load a separate test image
try
    testFeatures = pozymiai_raidems_atpazinti(imageFile, numRows);
catch exception
    fprintf('Error during test feature extraction: %s\n', exception.message);
    return;
end

% Convert test features to matrix
P_test = cell2mat(testFeatures);

% Ensure test data matches network input size
if size(P_test, 1) ~= size(P, 1)
    error('Test data size does not match training data size. Check feature extraction.');
end

% Simulate the network
Y_test = sim(net, P_test);

% Find the predicted digits (max value in each column corresponds to the digit)
[~, predictedDigits] = max(Y_test);

%% Step 5: Display Results
% Map the predictions to digits and display
fprintf('Recognized digits: ');
disp(predictedDigits - 1); % Subtract 1 to match digits (0-9)

%% Optional: Evaluate Performance
% If you have ground truth labels, compare them to predictions
% Example: Assuming ground truth is a sequence of 0-9 repeated
numTestSamples = size(P_test, 2);
groundTruth = repmat(0:9, 1, numTestSamples / numDigits);
accuracy = sum(predictedDigits - 1 == groundTruth) / numTestSamples * 100;
fprintf('Recognition Accuracy: %.2f%%\n', accuracy);
