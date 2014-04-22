train_data = load('data/train_small.mat');
test_data = load('data/test.mat');
train_small = create_sparse_img(double(train_data.train{5}.images));
test_images = create_sparse_img(double(test_data.test.images));
[w1, w2, err1, err2] = train_nn(train_small, double(train_data.train{5}.labels));
predictions1 = make_predictions(w1, [test_images ones(size(test_images,1), 1)]);
predictions2 = make_predictions(w2, [test_images ones(size(test_images,1), 1)]);
%meansqr_err(predictions, class_to_vector(test_data.test.labels))
class_predictions1 = zeros(size(predictions1,1), 1);
class_predictions2 = zeros(size(predictions2,1), 1);
for i=1:size(predictions,1)
   class_predictions1(i) = find(predictions1(i,:) == max(predictions1(i,:))) - 1;
   class_predictions2(i) = find(predictions2(i,:) == max(predictions2(i,:))) - 1;
end
test_err1 = double(sum(class_predictions1 ~= test_data.test.labels)) / double(length(class_predictions1))
test_err2 = double(sum(class_predictions2 ~= test_data.test.labels)) / double(length(class_predictions2))

figure;
title('Mean squared error on training set vs. epochs');
plot(1:length(err1), err1);
figure;
title('Cross entropy error on training set vs. epochs');
plot(1:length(err2), err2);