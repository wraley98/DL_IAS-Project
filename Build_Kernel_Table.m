function [TestData, truNetAcc, kernels] = Build_Kernel_Table

% Size of all kernels
H = 5;
% Create kernels
kernels = IAS_create_kernels(H);

% number of training images to be used in each test
numTrainImgs = [750 650 550 450 350 250 150];
% number of test to run
numTests = 10;

% train the initial neural networks (nn)
for ii = 1:length(numTrainImgs)

    % reset random number generator for each nn
    rng default

    % train nn
    [net,accuracy] = IAS_CNN_digits(H, numTrainImgs(ii));

    % store accuracy results and the nn
    truNetAcc(ii) = accuracy;
    netList(ii) = net;

end

% Loop controls what weights will be updated
% 1 -- First Layer // 2 -- Second Layer // 3 -- First then Second
for ww = 1:3

    % Convolution layer will not be trained in training process
    WeightLearnRateFactor = 0;

    % First loop will not allow for weight alterations in the convolution
    % layer, but the second loop will
    for kk = 1:2

        % reset the random number generator
        rng default

        % do not scale kernels
        scale = false;

        % run ten non-scaled tests
        for jj = 1:numTests
            
            [kk jj scale]
            
            % run the test for all nn  
            for ii = 1:length(numTrainImgs)

                [netb,accuracy,imdsValidation,indexes] = ...
                    IAS_CNN_digits_best_kernel(netList(ii),H,kernels, ...
                    numTrainImgs(ii), scale, WeightLearnRateFactor, ...
                    ww);
                
                % save individual test data
                accuracy_NonScaled(ii).acc = accuracy;
                indexes_NonScaled(ii).ind = indexes;

            end
            
            % save all test data into a structure
            nonScaled_Test(jj).Acc = accuracy_NonScaled;
            nonScaled_Test(jj).Ind = indexes_NonScaled;
        end
        
        % allow kernels to be scaled
        scale = true;
        % reset random number generator
        rng default
        
        % run ten non-scaled tests
        for jj = 1:numTests

            [kk jj scale]

            % run the test for all nn 
            for ii = 1:length(numTrainImgs)

                [netb,accuracy,imdsValidation,indexes] = ...
                    IAS_CNN_digits_best_kernel(netList(ii),H,kernels, ...
                    numTrainImgs(ii), scale, WeightLearnRateFactor, ww);
                
                % save individual test data
                accuracy_Scaled(ii).acc = accuracy;
                indexes_Scaled(ii).ind = indexes;
            end
            
            % save all test data into struct
            scaled_Test(jj).Acc = accuracy_Scaled;
            scaled_Test(jj).Ind = indexes_Scaled;

        end
        
        % save both scaled and non-scaled into struct
        TestData(kk).scaled = scaled_Test;
        TestData(kk).nonScaled = nonScaled_Test;
        
        % allow the next loop of conv layers to be trained
        WeightLearnRateFactor = 1;

    end
    
    % save the layer tests data into struct
    ConvLayerTest(ww).TestData = TestData;
    ConvLayerTest(ww).truNetAcc = truNetAcc;

end

save('TestData.mat', 'ConvLayerTest', 'truNetAcc', 'kernels');
