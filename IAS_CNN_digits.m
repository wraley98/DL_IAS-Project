function [net,accuracy,imdsValidation] = IAS_CNN_digits(H, numTrainFiles)
% IAS_CNN_digits - learn to classify 10 hand-written digits
% see https://www.mathworks.com/solutions/deep-learning/examples/training-a-model-from-scratch.html
% On input:
%     H (int): filter size (to get HxH)
% On output:
%     net (neural net structure): net trained to classify digits
%     accuracy (float): classification accuracy
%     imdsValidation (Validation stucture): validation set
% Call:
%     [net,a,val] = IAS_CNN_digits(5);
% Author:
%     T. Henderson
%     UU
%     Spring 2025
%

% create image data store
digitDatasetPath = fullfile(matlabroot,"toolbox","nnet","nndemos", ...
    "nndatasets","DigitDataset");
imds = imageDatastore(digitDatasetPath, ...
    IncludeSubfolders=true,LabelSource="foldernames");

% divide into training and validation sets
[imdsTrain,imdsValidation] = splitEachLabel(imds,...
    numTrainFiles,"randomize");

% create network layers
layers = [
    imageInputLayer([28 28 1])
	
    convolution2dLayer(H,4,Padding="same")
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

net = trainNetwork(imdsTrain,layers,options);

% Check accuracy
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);

tch = 0;
