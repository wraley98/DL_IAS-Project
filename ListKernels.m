function ListKernels

load("TestData_1.mat");

for jj = 1:3

    numTests = size(ConvLayerTest(jj).TestData(1).scaled(:), 1);
    numTrainImgTests = size(ConvLayerTest(jj).TestData(1).scaled(1).Acc(:), 1);
    numKern = size(ConvLayerTest(jj).TestData(1).scaled(1).Ind(1).ind(:), 1);

    for ii = 1:numTrainImgTests
        for tt = 1:numTests
            for kk = 1:numKern
                firstLayerTestKernelsScaled(tt, kk, ii) =...
                    ConvLayerTest(jj).TestData(1).scaled(tt).Ind(ii).ind(kk);

                firstLayerTestKernelsNonScaled(tt, kk, ii) =...
                    ConvLayerTest(jj).TestData(1).nonScaled(tt).Ind(ii).ind(kk);
            end
        end
    end

    firstLayerTestKernelsScaledMode = mode(firstLayerTestKernelsScaled);
    firstLayerTestKernelsNonScaledMode = mode(firstLayerTestKernelsNonScaled);

    w1(1,:) = firstLayerTestKernelsScaledMode(:,1,:);
    w1(2,:) = firstLayerTestKernelsNonScaledMode(:,1,:);

    w2(1,:) = firstLayerTestKernelsScaledMode(:,2,:);
    w2(2,:) = firstLayerTestKernelsNonScaledMode(:,2,:);

    w3(1,:) = firstLayerTestKernelsScaledMode(:,3,:);
    w3(2,:) = firstLayerTestKernelsNonScaledMode(:,3,:);

    w4(1,:) = firstLayerTestKernelsScaledMode(:,4,:);
    w4(2,:) = firstLayerTestKernelsNonScaledMode(:,4,:);

    xVals = 150:100:750;
    
    figure(jj)
    hold on

    subplot(2,2,1)
    bar(xVals, flipud(w1'))
    title("Weight 1 Mode")

    subplot(2,2,2)
    bar(xVals, flipud(w2'))
    title("Weight 2 Mode")

    subplot(2,2,3)
    bar(xVals, flipud(w3'))
    title("Weight 3 Mode")

    subplot(2,2,4)
    bar(xVals, flipud(w4'))
    title("Weight 4 Mode")
    
    hold off

end
i = 0;
