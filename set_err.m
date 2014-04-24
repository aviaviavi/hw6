function [ err ] = set_err(w, data, labels)

predictions = make_predictions(w, data);
out_layer = length(predictions);
class_predictions = zeros(size(predictions{out_layer},1), 1);
for i=1:size(class_predictions,1)
    class_predictions(i) = find(predictions{out_layer}(i,:) == max(predictions{out_layer}(i,:))) - 1;
end
err = double(sum(class_predictions ~= labels)) / double(length(class_predictions));

end