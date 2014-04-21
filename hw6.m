train_data = load('data/train_small.mat');
test_data = load('data/test.mat');
train_small = create_sparse_img(double(train_data.train{5}.images));
test_images = create_sparse_img(double(test_data.test.images));
w = train_nn(train_small, double(train_data.train{5}.labels));
predictions = make_predictions(w, [test_images ones(size(test_images,1), 1)]);
%meansqr_err(predictions, class_to_vector(test_data.test.labels))
class_predictions = zeros(size(predictions,1), 1);
for i=1:size(predictions,1)
   class_predictions(i) = find(predictions(i,:) == max(predictions(i,:))) - 1;
end
err = double(sum(class_predictions ~= test_data.test.labels)) / double(length(class_predictions));