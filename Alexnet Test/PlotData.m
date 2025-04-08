function PlotData

load("AlexnetTestData.mat");

figure(1)
hold on
fprintf("First Conv Layer of Alexnet used\n")
PrintData(firstLayerTestData, truNetAcc);
hold off

figure(2)
hold on
fprintf("Second Conv Layer of Alexnet used\n")
PrintData(secondLayerTestData, truNetAcc);
hold off

end

function PrintData(TestData, truNetAcc)

numTrainImgs = [750 650 550 450 350 250 150];
numTest = 1;

for ii = 1:length(numTrainImgs)

    scaledNoWeightChangeCurrIndex = [];
    nonScaledNoWeightChangeCurrIndex = [];
    scaledWeightChangeCurrIndex = [];
    nonScaledWeightChangeCurrIndex = [];

    for jj = 1:numTest

        scaledNoWeightChangeCurrIndex(jj) = ...
            TestData(1).scaled(jj).Acc(ii).acc;

        nonScaledNoWeightChangeCurrIndex(jj) = ...
            TestData(1).nonScaled(jj).Acc(ii).acc;

        scaledWeightChangeCurrIndex(jj) = ...
            TestData(2).scaled(jj).Acc(ii).acc;

        nonScaledWeightChangeCurrIndex(jj) = ....
            TestData(2).nonScaled(jj).Acc(ii).acc;

    end

    scaledNoWeightChange(ii) = mean(scaledNoWeightChangeCurrIndex);
    nonScaledNoWeightChange(ii) = mean(nonScaledNoWeightChangeCurrIndex);
    scaledWeightChange(ii) = mean(scaledWeightChangeCurrIndex);
    nonScaledWeightChange(ii) = mean(nonScaledWeightChangeCurrIndex);

end


scaledNoWeightChangeAvg = mean(scaledNoWeightChange);
nonScaledNoWeightChangeAvg = mean(nonScaledNoWeightChange);
scaledWeightChangeAvg = mean(scaledWeightChange);
nonScaledWeightChangeAvg = mean(nonScaledWeightChange);

scaledNoWeightChangeStd = std(scaledNoWeightChange);
nonScaledNoWeightChangeStd = std(nonScaledNoWeightChange);
scaledWeightChangeStd = std(scaledWeightChange);
nonScaledWeightChangeStd = std(nonScaledWeightChange);

scaledNoWeightChangeErr = mean(round(...
    (scaledNoWeightChange - truNetAcc)./ truNetAcc * 100 , 2));
nonScaledNoWeightChangeErr = mean(round(...
    (nonScaledNoWeightChange - truNetAcc) / truNetAcc * 100, 2));
scaledWeightChangeErr = mean(round((scaledWeightChange - truNetAcc) ...
    ./ truNetAcc * 100 , 2));
nonScaledWeightChangeErr = mean(round( ...
    (nonScaledWeightChange - truNetAcc) ./ truNetAcc * 100, 2));

tableData = [scaledNoWeightChangeAvg , nonScaledNoWeightChangeAvg, ...
    scaledWeightChangeAvg, nonScaledWeightChangeAvg;
    scaledNoWeightChangeStd, nonScaledNoWeightChangeStd,...
    scaledWeightChangeStd, nonScaledWeightChangeStd;...
    scaledNoWeightChangeErr, nonScaledNoWeightChangeErr, ...
    scaledWeightChangeErr, nonScaledWeightChangeErr];

rowNames = ["scaledNoWeightChange", "nonScaledNoWeightChange", ...
    "scaledWeightChange", "nonScaledWeightChange"];

T = table(rowNames', tableData(1,:)' , tableData(2,:)', tableData(3,:)' );

T.Properties.VariableNames = ["Test Name", "Average Accuracy", ...
    "Standard Deviation", "Average Error"];

disp(T)

plot(numTrainImgs, truNetAcc)
plot(numTrainImgs, scaledNoWeightChange)
plot(numTrainImgs , nonScaledNoWeightChange)
plot(numTrainImgs, scaledWeightChange)
plot(numTrainImgs , nonScaledWeightChange)

legend("trueNetAcc","scaledNoWeightChange", "nonScaledNoWeightChange", ...
    "scaledWeightChange", "nonScaledWeightChange")
title("Mean Accuracy of Neural Network vs Number of Training Images Used")
xlabel("Number of Training Images")
ylabel("Mean Accuracy of Trained Neural Network")

T = table(numTrainImgs', truNetAcc', scaledNoWeightChange',...
    nonScaledNoWeightChange',scaledWeightChange',nonScaledWeightChange');

T.Properties.VariableNames = ["numTrainImgs", "truNetAcc", "scaledNoWeightChange", ...
    "nonScaledNoWeightChange", "scaledWeightChange","nonScaledWeightChange"];

disp(T)
end

