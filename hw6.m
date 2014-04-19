train_data = load('data/train_small.mat');
test_data = load('data/test.mat');
train_small = create_sparse_img(double(train_data.train{2}.images));
test_images = create_sparse_img(double(test_data.test.images));
[w, b] = train_nn(train_small, double(train_data.train{2}.labels));
predictions = make_predictions(w, b, test_images);
%meansqr_err(predictions, class_to_vector(test_data.test.labels))
class_predictions = zeros(size(predictions,1), 1);
for i=1:size(predictions,1)
   class_predictions(i) = find(predictions(i,:) == max(predictions(i,:))) - 1;
end
err = double(sum(class_predictions ~= test_data.test.labels)) / double(length(class_predictions));