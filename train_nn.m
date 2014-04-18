function [ w, b ] = train_nn( obs, out, sizes_hidden, error_func)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    if nargin < 3
        sizes_hidden = [];
        error_func = 1;
    end
    og_step_size = .01;
    step_size = .01;
    w = {};
    % 10 is the number of output classes
    sizes = [size(obs, 2) sizes_hidden 10];
    b = {};
    for i = 1:length(sizes) - 1
       w{i} = rand(sizes(i), sizes(i+1));
       b{i} = rand(1, sizes(i+1));
    end
    
    p = randperm(size(obs));
    rand_obs = reshape(obs(p,:), size(obs, 1), size(obs, 2));
    rand_out = reshape(out(p,:), size(out, 1), size(out, 2));
    
    num_batches = ceil(size(obs, 1) / 200);
    num_epocs = 500;
    for i = 1:num_epocs
        i
        step_size = (1.0 / (i ^ .5)) * og_step_size;
        for j = 1:num_batches
            start = (j-1) * 200 + 1;
            stop = min(start + 199, size(rand_obs, 1));
            curr_set = rand_obs(start:stop,:);
            predictions = make_predictions(w, b, curr_set);
            truths = class_to_vector(rand_out(start:stop));
            if error_func == 1
                error = meansq_err(predictions, truths);
                deltas = -sum((truths - predictions) .* predictions .* (1 - predictions));
                gradients = zeros(size(curr_set, 2), size(truths, 2));
                for data = 1:size(truths, 1)
                    for k = 1:size(truths, 2)
                        d = (truths(data, k) - predictions(data, k)) .* predictions(data, k) .* (1 - predictions(data, k));
                        for jay = 1:size(rand_obs, 2)
                            gradients(jay, k) = gradients(jay, k) + (d * curr_set(data, jay));
                        end
                    end
                end
            else %cross entropy
                error = cross_entropy(predictions, truths);
                deltas = sum((((1 - truths) ./ (1 - predictions)) - (truths ./ predictions)) .* predictions .* (1 - predictions));
            end
            %update weights and biases, last  
            w{1} = w{1} - (step_size * gradients);
            %b{1} = b{1} - (step_size * deltas);
            %w{length(w)} = w{length(w)} - step_size * 
        end
    end
    
end

