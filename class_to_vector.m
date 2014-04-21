function [ output ] = class_to_vector( class_nums, num_out )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    output = zeros(size(class_nums, 1), num_out);
    for i = 1:length(class_nums)
        output(i, class_nums(i) + 1) = 1;
    end
end

