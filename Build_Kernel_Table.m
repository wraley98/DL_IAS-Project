function [TestData, truNetAcc, kernels] = Build_Kernel_Table

H = 5;
kernels = IAS_create_kernels(H);

numTrainImgs = [750];% 650 550 450 350 250 150];
numTests = 10;
WeightLearnRateFactor = 0;



for ii = 1:length(numTrainImgs)
    rng default
    
    [net,accuracy,imdsValidation] = IAS_CNN_digits(H, numTrainImgs(ii));
    truNetAcc(ii) = accuracy;
    netList(ii) = net;

end

for kk = 1:2
    rng default
    for jj = 1:numTests

        scale = false;

        [kk jj numTest]

        for ii = 1:length(numTrainImgs)
            


            [netb,accuracy,imdsValidation,indexes] = ...
                IAS_CNN_digits_best_kernel(netList(ii),H,kernels, ...
                numTrainImgs(ii), scale, WeightLearnRateFactor);

            accuracy_NonScaled(ii).acc = accuracy;
            indexes_NonScaled(ii).ind = indexes;

        end

        nonScaled_Test(jj).Acc = accuracy_NonScaled;
        nonScaled_Test(jj).Ind = indexes_NonScaled;

        scale = true;

        for ii = 1:length(numTrainImgs)

            [netb,accuracy,imdsValidation,indexes] = ...
                IAS_CNN_digits_best_kernel(netList(ii),H,kernels, ...
                numTrainImgs(ii), scale, WeightLearnRateFactor);

            accuracy_Scaled(ii).acc = accuracy;
            indexes_Scaled(ii).ind = indexes;
        end

        scaled_Test(jj).Acc = accuracy_Scaled;
        scaled_Test(jj).Ind = indexes_Scaled;

        TestData(kk).scaled = scaled_Test;
        TestData(kk).nonScaled = nonScaled_Test;

    end

    WeightLearnRateFactor = 1;


end

save('TestDataTest.mat', 'TestData', 'truNetAcc', 'kernels');