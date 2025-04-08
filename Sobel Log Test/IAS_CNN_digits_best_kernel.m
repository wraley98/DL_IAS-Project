function [netb,accuracy,imdsValidation] = IAS_CNN_digits_best_kernel(H,....
                net, kernels,numTrainImgs, scale, WeightLearnRateFactor)
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

W = IAS_extract_weights(net,2);

sizeArr = size(W);

if scale
    for ii = 1:4

        kernels(ii).kernel = IAS_scale( W(:,:,1,ii), kernels(ii).kernel);

    end
end

W = IAS_K2W(kernels , sizeArr);

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

    convolution2dLayer(H,4,'Weights',W,...
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

end
