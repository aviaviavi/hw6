function [ predicted ] = make_predictions( w, obs )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    i = 0;
    if length(w) ~= 1
        for i = 1:length(w) - 1
         %obs = tanh(obs * w{i});
        end
    end
    predicted = sigmoid(obs * w{i+1});
end

