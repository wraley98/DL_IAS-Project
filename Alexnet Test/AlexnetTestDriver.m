function AlexnetTestDriver


aNet = alexnet;

% use alexnet first conv layer for testing
[firstLayerTestData, truNetAcc, kernels] = Build_Kernel_Table(11, 2, aNet);

% use alexnet second conv layer for testing
[secondLayerTestData, ~, ~] = Build_Kernel_Table(3 , 10, aNet);

save('AlexnetTestData.mat', 'firstLayerTestData', ...
            'secondLayerTestData', 'kernels', 'truNetAcc');