function [ w1, w2, err1, err2 ] = train_nn(obs, out, sizes_hidden, error_func)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    if nargin < 3
        sizes_hidden = [];
    end
    og_step_size = .0000001;
    % add in bias term
    obs = [obs ones(size(obs,1),1)];
    num_in = size(obs,2);
    % 10 is the number of output classes
    num_out = 10;
    sizes = [num_in sizes_hidden num_out];
    w1 = {};
    w2 = {};
    for i = 1:length(sizes) - 1
        % center weights around 0
        w1{i} = rand(sizes(i), sizes(i+1)) * .0010 - .0005;
        w2{i} = w1{i};
    end
    num_batches = ceil(size(obs, 1) / 200);
    num_epochs = 400;
    err1 = zeros(1, num_epochs / 10 + 1);
    err2 = zeros(1, num_epochs / 10 + 1);
    for epoch = 1:num_epochs
        epoch
        % shuffle data
        p = randperm(size(obs, 1));
        rand_obs = reshape(obs(p,:), size(obs, 1), size(obs, 2));
        rand_out = reshape(out(p,:), size(out, 1), size(out, 2));
        step_size = (1.0 / ((epoch+1) ^ .5)) * og_step_size;
        % minibatch update
        for batch=1:num_batches
            % prepare data
            batch_size = min(200, size(rand_obs,1) - (batch - 1)*200);
            start = (batch - 1)*200 + 1;
            stop = start + batch_size - 1;
            data = rand_obs(start:stop,:); % batch_size x num_in
            labels = class_to_vector(rand_out(start:stop), num_out); % batch_size x num_out
            % make predictions and update for output layer
            predictions1 = make_predictions(w1, data); % batch_size x num_out
            predictions2 = make_predictions(w2, data); % batch_size x num_out
            % store error rates to plot
            if (epoch == 1 || mod(epoch,10) == 0)
               err1(floor(epoch/10)+1) = meansq_err(labels, predictions1);
               err2(floor(epoch/10)+1) = cross_ent_err(labels, predictions2);
            end
            gradients1 = zeros(size(w1{1}));
            deltas1 = zeros(1,num_out);
            gradients2 = zeros(size(w2{1}));
            deltas2 = zeros(1,num_out);
            for datum=1:batch_size
                delta1 = -(labels(datum,:) - predictions1(datum,:)) .* predictions1(datum,:) .* (1 - predictions1(datum,:)); % 1 x num_out
                deltas1 = deltas1 + delta1;
                delta2 = ((1 - labels(datum,:)) ./ (1 - predictions2(datum,:)) - labels(datum,:) ./ predictions2(datum,:)) .*  predictions2(datum,:) .* (1 - predictions2(datum,:)); % 1 x num_out
                deltas2 = deltas2 + delta2;
                gradients1 = gradients1 + (data(datum,:)' * delta1);
                gradients2 = gradients2 + (data(datum,:)' * delta2);
            end
            w1{1} = w1{1} - step_size * gradients1;
            w2{1} = w2{1} - step_size * gradients2;
        end
    end
end

