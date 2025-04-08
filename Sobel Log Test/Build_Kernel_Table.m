function [TestData, truNetAcc] = Build_Kernel_Table

% Size of all kernels
H = 5;
% Create kernels
[Sobelkernels , Logkernels] = IAS_create_kernels(H);

kernelStruct(1).type = Sobelkernels;
kernelStruct(2).type = Logkernels;

% number of training images to be used in each test
numTrainImgs = [750 650 550 450 350 250 150];
% number of test to run
numTests = 3;

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

% Convolution layer will not be adjusted in training process
WeightLearnRateFactor = 0;

for kernelNum = 1:2
     
    kernels = kernelStruct(kernelNum).type;

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

            for ii = 1:length(numTrainImgs)

                [netb,accuracy] = ...
                    IAS_CNN_digits_best_kernel(H,netList(ii), kernels, ...
                    numTrainImgs(ii), scale, WeightLearnRateFactor);

                accuracy_NonScaled(ii) = accuracy;

            end

            nonScaled_Test(jj).acc = accuracy_NonScaled;
        end

        scale = true;
        rng default

        for jj = 1:numTests

            [kk jj scale]

            for ii = 1:length(numTrainImgs)

                 [netb,accuracy] = ...
                    IAS_CNN_digits_best_kernel(H, netList(ii), kernels, ...
                    numTrainImgs(ii), scale, WeightLearnRateFactor);

                accuracy_Scaled(ii) = accuracy;
            end

            scaled_Test(jj).acc = accuracy_Scaled;

        end

        TestData(kk).scaled = scaled_Test;
        TestData(kk).nonScaled = nonScaled_Test;

        WeightLearnRateFactor = 1;

    end
    
    kernelTest(kernelNum).type = TestData;

end

save('SobelvLogTestData.mat', 'kernelTest', 'truNetAcc', 'kernels');
