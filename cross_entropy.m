function [ error ] = cross_entropy( true, predicted )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    error = sum(sum(true .* log(predicted + .001) + (1 - true) .* log(1.001 - predicted)));
end

