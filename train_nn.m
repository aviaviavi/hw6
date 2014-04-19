function [ w, b ] = train_nn(obs, out, sizes_hidden, error_func)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    if nargin < 3
        sizes_hidden = [];
        error_func = 1;
    end
    og_step_size = .1;
    w = {};
    % 10 is the number of output classes
    sizes = [size(obs, 2) sizes_hidden 10];
    b = {};
    for i = 1:length(sizes) - 1
       w{i} = rand(sizes(i), sizes(i+1));
       b{i} = rand(1, sizes(i+1));
    end
    num_batches = ceil(size(obs, 1) / 200);
    num_epochs = 200;
    for epoch = 1:num_epochs
        % shuffle data
        p = randperm(size(obs, 1));
        rand_obs = reshape(obs(p,:), size(obs, 1), size(obs, 2));
        rand_out = reshape(out(p,:), size(out, 1), size(out, 2));
        step_size = (1.0 / (i ^ .5)) * og_step_size;
        % minibatch update
        for batch=1:num_batches
            % prepare data
            batch_size = min(200, size(rand_obs,1) - (batch - 1)*200);
            start = (batch - 1)*200 + 1;
            stop = start + batch_size - 1;
            data = rand_obs(start:stop,:); % batch_size x 784
            labels = class_to_vector(rand_out(start:stop)); % batch_size x 10
            % make predictions and update for output layer
            predictions = make_predictions(w, b, data); % batch_size x 10
            gradients = zeros(size(w{1}));
            deltas = zeros(1,10);
            for datum=1:batch_size
                delta = -(labels(datum,:) - predictions(datum,:)) .* predictions(datum,:) .* (1 - predictions(datum,:)); % 1 x 10
                deltas = deltas + delta;
                gradients = gradients + data(datum,:)' * delta;
            end
            w{1} = w{1} - step_size * gradients;
            b{1} = b{1} - step_size * deltas;
        end
    end     
end

