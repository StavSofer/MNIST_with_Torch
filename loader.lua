require 'nn'
require 'cunn'
require 'optim'
local mnist = require 'mnist';

--[[
    Function that gets as input a test set and labels
    loads a trained network (model) and returns its error
]]

function testError(data, labels)
    -- returns testLoss, testError, confusion

    --local vars
    local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    local lossAcc = 0
    local numBatches = 0
    local batchSize = 128
    -- our criterion
    local criterion = nn.ClassNLLCriterion():cuda()

    -- forward inputs (with step=batchsize)
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        -- narrow data/labels
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        -- forward
        local y = model:forward(x)
        -- get error
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        -- update confusion matrix
        confusion:batchAdd(y,yt)
    end
    
    confusion:updateValids()
    -- compute average loss and error
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid

    return avgLoss, avgError, tostring(confusion)
end

-- train data is needed to center the test set
local trainData = mnist.traindataset().data:float();
local trainLabels = mnist.traindataset().label:add(1);
-- convert
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);
-- get mean/std from train
local mean = trainData:mean()
local std = trainData:std()
-- center test set (with train set)
testData:add(-mean):div(std);

-- load model and invoke function
-- print error/loss
model = torch.load('mymodel')

-- create tensor to assign loss/err
loss = torch.Tensor(1)
err = torch.Tensor(1)

-- compute and print error using our function
loss, err, confusion = testError(testData,testLabels)

print('Test error: ' .. err, 'Test Loss: ' .. loss)

