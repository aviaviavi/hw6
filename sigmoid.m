function [ a ] = sigmoid( z )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
    %credit to stanford for this one
    a = 1.0 ./ (1.0 + exp(-z));
end

