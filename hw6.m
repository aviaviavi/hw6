train_data = load('data/train_small.mat');
test_data = load('data/test.mat');
train_small = create_sparse_img(double(train_data.train{2}.images));
test_images = create_sparse_img(double(test_data.test.images));
[w1, w2, test_err1, test_err2, train_err1, train_err2] = train_nn(train_small, double(train_data.train{2}.labels), [300 100]);
predictions1 = make_predictions(w1, [test_images ones(size(test_images,1), 1)]);
predictions2 = make_predictions(w2, [test_images ones(size(test_images,1), 1)]);

figure;
plot(1:10:length(train_err1) * 10, train_err1, 1:10:(length(test_err1) * 10), test_err1);
legend('training error', 'test error');
title('Error MSE vs. epochs');
figure;
plot(1:10:length(train_err2) * 10, train_err2, 1:10:(length(test_err2) * 10), test_err2);
legend('training error', 'test error');
title('Cross entropy error vs. epochs');
