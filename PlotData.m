function PlotData

load("TestData.mat");

numTrainImgs = [750 650 550 450 350 250 150];
numTest = 10;

for ii =1:length(numTrainImgs)

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

figure(1)

hold on

plot(numTrainImgs, truNetAcc)
plot(numTrainImgs, scaledNoWeightChange)
plot(numTrainImgs , nonScaledNoWeightChange)
plot(numTrainImgs, scaledWeightChange)
plot(numTrainImgs , nonScaledWeightChange)

hold off

legend("trueNetAcc","scaledNoWeightChange", "nonScaledNoWeightChange", ...
    "scaledWeightChange", "nonScaledWeightChange")

T = table(numTrainImgs', truNetAcc', scaledNoWeightChange',...
    nonScaledNoWeightChange',scaledWeightChange',nonScaledWeightChange');

T.Properties.VariableNames = ["numTrainImgs", "truNetAcc", "scaledNoWeightChange", ...
    "nonScaledNoWeightChange", "scaledWeightChange","nonScaledWeightChange"];

disp(T)
