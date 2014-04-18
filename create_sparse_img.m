% Justin Kay, SID 21762610

function [newImages] = create_sparse_img(images)

% concatenates all 2d image examples into 1d vectors so that the matrix of
% image examples is 2d, converts that to a sparse matrix

newImages = zeros(size(images,3),784);

for i = 1:size(images,3)
    concatted = images(:,:,i);
    for j = 1:784;
        newImages(i,j) = concatted(j);
    end
    newImages(i,:) = newImages(i,:) / sum(newImages(i,:));
end

newImages = sparse(newImages);

end