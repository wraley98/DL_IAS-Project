function [netb,accuracy,imdsValidation,indList] = ...
    IAS_CNN_digits_best_kernel(net,H,kernels,numTrainImgs,...
                                    scale,WeightLearnRateFactor)
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

% determine number of kernels
num_kernels = length(kernels);

% extract the weights from the second conv layer
W = IAS_extract_weights(net,6);

% create kernels from the weights
[W, sizeArr] = IAS_W2K(W);

% num of channels * num of layers
numWeights = sizeArr(3) * sizeArr(4);

% initialize the error and index list
errList = inf(1 , numWeights);
indList = zeros(1,numWeights);

% compare each kernel
for kk = 1:num_kernels

    % kernel
    kern = kernels(kk).w;

    % compare the specified kernel to each weight
    for jj = 1:numWeights
        
        % scale if necessary
        if scale
            kern = IAS_scale(double(W(jj).w) , double(kern));
        end
        
        % determine the average difference between the weight and kernel
        err = mean(mean(abs(double(kern)-double(W(jj).w))));
        
        % update the best kernel if necessary
        if err<errList(jj)
            errList(jj) = err;
            indList(jj) = kk;
        end
    end
end

% loop through and and add each of the best kernels 
for jj = 1:numWeights
    
    % select the kernel
    kern =  kernels(indList(jj)).w;
    
    % scale if necessary
    if scale
        kern = IAS_scale(W(jj).w,kern);
    end
    
    % add it to the best kenrel struct
    bestKernels(jj).kernel = kern;

end

% create weights from the kernel
weights = IAS_K2W(bestKernels, sizeArr);

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

    convolution2dLayer(H,4,Padding="same")
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,Stride=2)

    convolution2dLayer(H,4,'Weights',weights,...
    'WeightLearnRateFactor',WeightLearnRateFactor,Padding="same")
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

% train network
netb = trainNetwork(imdsTrain,layers,options);

% Check accuracy
YPred = classify(netb,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);
