function [netb,accuracy,imdsValidation,indList] = ...
    IAS_CNN_digits_best_kernel(net,H,kernels,numTrainImgs...
    ,scale,WeightLearnRateFactor, convLayer)
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

% determine number of kernels that will be compared
num_kernels = length(kernels);

% get the weights for both conv layers
layerWeight(1).weights= IAS_extract_weights(net,2);
layerWeight(2).weights= IAS_extract_weights(net,6);

if convLayer == 1 || convLayer == 2
    % retrieve the specified conv layer weight
    weight = layerWeight(convLayer).weights;
else
    % if the both conv layers are being trained, train the first
    % layer initially
    % call this function again for only the first layer
    [net,~,~,indListFirstLayer] = ...
        IAS_CNN_digits_best_kernel(net,H,kernels,numTrainImgs,scale,...
        WeightLearnRateFactor, 1);
    % retrieve the new weights
    weight = IAS_extract_weights(net,6);
end

% determine the size of the layer and retrieve the weights
[W, sizeArr] = IAS_W2K(weight);

% num of channels * num of layers
numWeights = sizeArr(3) * sizeArr(4);

% initialize the error list and the index list
errList = inf(1 , numWeights);
indList = zeros(1,numWeights);

% compare all kernels
for k = 1:num_kernels

    % retrieve kernel 
    K = kernels(k).kernel;
    
    % compare the kernel to all weights
    for jj = 1:numWeights
        
        % scale if necessary
        if scale
            K = IAS_scale(double(W(jj).w) , double(K));
        end
        
        % determine the average difference between the kernel and the
        % weight
        err = mean(mean(abs(double(K)-double(W(jj).w))));
        
        % update the best kernel if necessary
        if err<errList(jj)
            errList(jj) = err;
            indList(jj) = k;
        end
    end
end

% loop through and update the weights with the closest kernels
for jj = 1:numWeights

    k =  kernels(indList(jj)).kernel;
    
    % scale the kernel if necessary
    if scale
        k = IAS_scale(W(jj).w,k);
    end
    
    % add kernel to the best kernel struct for replacement
    bestKernels(jj).kernel = k;

end

% if both conv layers are being trained, add both layer index to the index
% list
if convLayer == 3
    indStruct(1).ind = indListFirstLayer;
    indStruct(2).ind = indList;

    indList = indStruct;
end

% create weights from the selected kernels
weight = IAS_K2W(bestKernels,sizeArr);


% create image data store
digitDatasetPath = fullfile(matlabroot,"toolbox","nnet","nndemos", ...
    "nndatasets","DigitDataset");
imds = imageDatastore(digitDatasetPath, ...
    IncludeSubfolders=true,LabelSource="foldernames");

% divide into training and validation sets
[imdsTrain,imdsValidation] = ...
    splitEachLabel(imds,numTrainImgs,"randomize");

% create network layers depending on which conv layer is to be updated
if convLayer == 1
    layers = UpdateFirstConvLayers(weight, H, WeightLearnRateFactor);
else
    layers = UpdateSecondConvLayers(weight, H, WeightLearnRateFactor);
end

% train the network
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.01, ...
    MaxEpochs=8, ...
    Plots="none");

% train network
netb = trainNetwork(imdsTrain,layers,options);

% Check accuracy
YPred = classify(netb,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);

end

% create layers with the first convolution layer being updated 
function layers = UpdateFirstConvLayers(weight, H,...
    WeightLearnRateFactor)

layers = [
    imageInputLayer([28 28 1])

    convolution2dLayer(H,4,'Weights',weight,...
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
end

% create layers with the second convolution layer being updated 
function layers = UpdateSecondConvLayers(weight, H,...
    WeightLearnRateFactor)

layers = [
    imageInputLayer([28 28 1])

    convolution2dLayer(H,4,Padding="same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,Stride=2)

    convolution2dLayer(H,4,'Weights',weight,...
    'WeightLearnRateFactor',WeightLearnRateFactor,Padding="same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,Stride=2)

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
    ];
end
