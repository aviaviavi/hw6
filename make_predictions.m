function [ predicted ] = make_predictions( w, obs )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    predicted = {obs};
    i = 0;
    if length(w) ~= 1
        % hidden layers
        for i = 1:length(w) - 1
            predicted{i+1} = tanh(predicted{i} * w{i});
        end
    end
    % output layer
    predicted{i+2} = sigmoid(predicted{i+1} * w{i+1});
end

