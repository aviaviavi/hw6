function [ error ] = meansq_err( true, predicted )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    error = sum((1/size(true, 1)) * sum((true - predicted) .^ 2));
end