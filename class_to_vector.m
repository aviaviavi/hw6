function [ output ] = class_to_vector( class_nums )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    output = zeros(length(class_nums), 10);
    for i = 1:length(class_nums)
        output(i, class_nums(i) + 1) = 1;
    end
end

