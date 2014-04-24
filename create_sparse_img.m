% Justin Kay, SID 21762610

function [newImages] = create_sparse_img(images)

% concatenates all 2d image examples into 1d vectors so that the matrix of
% image examples is 2d, converts that to a sparse matrix

newImages = zeros(size(images,3),784);

for i = 1:size(newImages,1)
    concatted = images(:,:,i);
    for j = 1:size(newImages,2);
        newImages(i,j) = concatted(j);
    end
end
means = mean(newImages);
for i=1:size(newImages,1)
    newImages(i,:) = newImages(i,:) - means;
end
for i=1:size(newImages,2)
    newImages(:,i) = newImages(:,i) / (std(newImages(:,i))+.0001);
end
newImages = sparse(newImages);

end