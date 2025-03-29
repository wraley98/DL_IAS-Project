function [netb,accuracy,imdsValidation,indList] = ...
    IAS_CNN_digits_best_kernel(net,H,kernels,numTrainImgs...
    ,scale,WeightLearnRateFactor)
% IAS_CNN_digits_best_kernel - find best matching kernels for weights in net
% On input:
%     net (Matlab neural net struct): base neural net with learned kernels
%     H (int): filter size (to get HxH)
%     kernels (kernels struct): set of standard image processing kernels
% On output:
%     netb (neural net structure): net trained using best matched kernels
%     accuracy (float): classification accuracy
%     imdsValidation (Validation stucture): validation set
%     indexes (1x4 vector): best match kernel indexes
% Call:
%     [netb,ab,valb,ind] = IAS_CNN_digits_best_kernel(net,5,kernels);
% Author:
%     T. Henderson
%     UU
%     Spring 2025
%


close all

num_kernels = length(kernels);

weights= IAS_extract_weights(net,2);

W(1).w = IAS_W2K(weights,1);
W(2).w = IAS_W2K(weights,2);
W(3).w = IAS_W2K(weights,3);
W(4).w = IAS_W2K(weights,4);

errList = [inf inf inf inf];

indList = zeros(1,4);

for k = 1:num_kernels

    % kernel
    K = kernels(k).kernel;

    for ii = 1:4

        if scale
            K = IAS_scale(double(W(ii).w) , double(K));
        end

        err = mean(mean(abs(double(K)-double(W(ii).w))));

        if err<errList(ii)
            errList(ii) = err;
            indList(ii) = k;
        end
    end
end

weights = zeros(5,5,1,4);

for ii = 1:4

    K = kernels(indList(ii)).kernel;

    if scale
        K = IAS_scale(W(ii).w,K);
    end
    
    weights = IAS_K2W(K,weights,ii);

end

% create image data store
digitDatasetPath = fullfile(matlabroot,"toolbox","nnet","nndemos", ...
    "nndatasets","DigitDataset");
imds = imageDatastore(digitDatasetPath, ...
    IncludeSubfolders=true,LabelSource="foldernames");

% divide into training and validation sets
[imdsTrain,imdsValidation] = ...
    splitEachLabel(imds,numTrainImgs,"randomize");

% create network layers
layers = [
    imageInputLayer([28 28 1])

    convolution2dLayer(H,4,'Weights',weights,...
    'WeightLearnRateFactor',WeightLearnRateFactor,Padding="same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,Stride=2)

    convolution2dLayer(H,4,Padding="same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,Stride=2)

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
    ];

% train the network
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.01, ...
    MaxEpochs=8, ...
    Plots="none");

netb = trainNetwork(imdsTrain,layers,options);

% Check accuracy
YPred = classify(netb,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);

tch = 0;