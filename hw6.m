train_data = load('data/train_small.mat');
test_data = load('data/test.mat');
train_small = create_sparse_img(double(train_data.train{3}.images));
test_images = create_sparse_img(double(test_data.test.images));
[w, b] = train_nn(train_small, double(train_data.train{3}.labels));
predictions = make_predictions(w, b, test_images);
%meansqr_err(predictions, class_to_vector(test_data.test.labels))
