function [TestData, truNetAcc, kernels] = ...
    Build_Kernel_Table(H , layer, aNet)

% retrieve alexnets specified conv layer
kernels = IAS_extract_weights(aNet,layer);

% update weights to usable kernels
[kernels, sizeArr] = IAS_W2K(kernels);

% number of training images to be used in each test
numTrainImgs = [750 650 550 450 350 250 150];

% number of test to run
numTests = 10;

% train the initial neural networks (nn)
for ii = 1:length(numTrainImgs)

    % reset random number generator for each nn
    rng default

    % train nn
    [net,accuracy,imdsValidation] = IAS_CNN_digits(H, numTrainImgs(ii));

    % store accuracy results and the nn
    truNetAcc(ii) = accuracy;
    netList(ii) = net;

end

% Convolution layer will not be trained in training process
WeightLearnRateFactor = 0;

% loop for controling train status on the conv kernels
for kk = 1:2

    % reset the random number generator
    rng default
    % do not scale kernels
    scale = false;

    % run non-scaled tests
    for jj = 1:numTests

        [kk jj scale]
        
        % test for each nn
        for ii = 1:length(numTrainImgs)

            [netb,accuracy,imdsValidation,indexes] = ...
                IAS_CNN_digits_best_kernel(netList(ii),H,kernels, ...
                numTrainImgs(ii), scale, WeightLearnRateFactor);
            
            % save nn data 
            accuracy_NonScaled(ii).acc = accuracy;
            indexes_NonScaled(ii).ind = indexes;

        end
        
        % save all test data to struct
        nonScaled_Test(jj).Acc = accuracy_NonScaled;
        nonScaled_Test(jj).Ind = indexes_NonScaled;
    end
    
    % allow scaling
    scale = true;
    % reset random number generator
    rng default
    
    % loop through scaled test
    for jj = 1:numTests

        [kk jj scale]
        
        % run tests for each nn
        for ii = 1:length(numTrainImgs)
           
            [netb,accuracy,imdsValidation,indexes] = ...
                IAS_CNN_digits_best_kernel(netList(ii),H,kernels, ...
                numTrainImgs(ii), scale, WeightLearnRateFactor);
            
            % save each nn data
            accuracy_Scaled(ii).acc = accuracy;
            indexes_Scaled(ii).ind = indexes;
        end
        
        % save the entire round of testing data to struct
        scaled_Test(jj).Acc = accuracy_Scaled;
        scaled_Test(jj).Ind = indexes_Scaled;

    end
    
    % save the training specified test data
    TestData(kk).scaled = scaled_Test;
    TestData(kk).nonScaled = nonScaled_Test;
    
    % allow for training
    WeightLearnRateFactor = 1;

end


