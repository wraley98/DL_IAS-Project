function PlotData

load("TestData.mat");

numTrainImgs = [750 650 550 450 350 250 150];

figure(1)
hold on
for ii =1:length(numTrainImgs)

    scaledNoWeightChange(ii) = TestData(1).scaled.Acc(ii).acc;
    nonScaledNoWeightChange(ii) = TestData(1).nonScaled.Acc(ii).acc;
    scaledWeightChange(ii) = TestData(2).scaled.Acc(ii).acc;
    nonScaledWeightChange(ii) = TestData(2).nonScaled.Acc(ii).acc;

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
