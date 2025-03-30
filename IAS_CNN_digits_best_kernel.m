function [netb,accuracy,imdsValidation,indList] = ...
    IAS_CNN_digits_best_kernel(net,H,kernels,numTrainImgs...
    ,scale,WeightLearnRateFactor, convLayerUpdate)
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

layerWeight(1).weights= IAS_extract_weights(net,2);
layerWeight(2).weights= IAS_extract_weights(net,6);

if convLayerUpdate < 3
    weightsCount = 1;
else
    weightsCount = 2;
end

for ii = 1:weightsCount

    if ~(convLayerUpdate == 3)
        weights =  layerWeight(convLayerUpdate).weights;
    else
        weights =  layerWeight(ii).weights;
    end

    [W, sizeArr] = IAS_W2K(weights);

    % num of channels * num of layers
    numWeights = sizeArr(3) * sizeArr(4);


    errList = inf(1 , numWeights);

    indList = zeros(1,numWeights);

    for k = 1:num_kernels

        % kernel
        K = kernels(k).kernel;

        for jj = 1:numWeights

            if scale
                K = IAS_scale(double(W(jj).w) , double(K));
            end

            err = mean(mean(abs(double(K)-double(W(jj).w))));

            if err<errList(jj)
                errList(jj) = err;
                indList(jj) = k;
            end
        end
    end

    for jj = 1:numWeights

        k =  kernels(indList(jj)).kernel;

        if scale
            k = IAS_scale(W(jj).w,k);
        end

        bestKernels(jj).kernel = k;

    end

    if weightsCount == 1
        layerWeight(convLayerUpdate).weights = ...
            IAS_K2W(bestKernels,sizeArr);
    else
        layerWeight(ii).weights = ...
            IAS_K2W(bestKernels,sizeArr);
    end
end

% create image data store
digitDatasetPath = fullfile(matlabroot,"toolbox","nnet","nndemos", ...
    "nndatasets","DigitDataset");
imds = imageDatastore(digitDatasetPath, ...
    IncludeSubfolders=true,LabelSource="foldernames");

% divide into training and validation sets
[imdsTrain,imdsValidation] = ...
    splitEachLabel(imds,numTrainImgs,"randomize");

% create network layers depending on which conv layer is to be updated
if convLayerUpdate == 1
    layers = UpdateFirstConvLayers(layerWeight, H, WeightLearnRateFactor);
elseif convLayerUpdate == 2
    layers = UpdateSecondConvLayers(layerWeight, H, WeightLearnRateFactor);
else
    layers = UpdateBothConvLayers(layerWeight, H, WeightLearnRateFactor);
end


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

function layers = UpdateFirstConvLayers(layerWeight, H,...
                                                WeightLearnRateFactor)

layers = [
    imageInputLayer([28 28 1])

    convolution2dLayer(H,4,'Weights',layerWeight(1).weights,...
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

function layers = UpdateSecondConvLayers(layerWeight, H,...
                                            WeightLearnRateFactor)

layers = [
    imageInputLayer([28 28 1])

    convolution2dLayer(H,4,Padding="same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,Stride=2)
    
  
    convolution2dLayer(H,4,'Weights',layerWeight(2).weights,...
    'WeightLearnRateFactor',WeightLearnRateFactor,Padding="same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,Stride=2)

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
    ];
end

function layers = UpdateBothConvLayers(layerWeight, H, ...
                                                    WeightLearnRateFactor)

layers = [
    imageInputLayer([28 28 1])

    convolution2dLayer(H,4,'Weights',layerWeight(1).weights,...
    'WeightLearnRateFactor',WeightLearnRateFactor,Padding="same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,Stride=2)
    
    convolution2dLayer(H,4,'Weights',layerWeight(2).weights,...
    'WeightLearnRateFactor',WeightLearnRateFactor,Padding="same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,Stride=2)

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
    ];
end