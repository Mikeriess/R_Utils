library(mxnet)

#Load custom MSE eval metric:
mx.metric.mse <- mx.metric.custom("mse", function(label, pred) {
  res <- mean((label-pred)^2)
  return(res)
})


library(mxnet)

data(BostonHousing, package="mlbench")
train.ind = seq(1, 506, 3)

train.x = data.matrix(BostonHousing[train.ind, -14])
train.y = BostonHousing[train.ind, 14]
test.x = data.matrix(BostonHousing[-train.ind, -14])
test.y = BostonHousing[-train.ind, 14]

data <- mx.symbol.Variable("data")      
label <- mx.symbol.Variable("label")

# Train network with a hidden layer
fc1 <- mx.symbol.FullyConnected(data, num_hidden=10, name="fc1")
act1 <- mx.symbol.Activation(fc1, act_type="tanh", name="act1") 
fc2 <- mx.symbol.FullyConnected(act1, num_hidden=1, name="fc2")
lro <- mx.symbol.LinearRegressionOutput(data=fc2, label=label, name="lro2")

mx.set.seed(0)


#Log and store model after each epoch:
logger <- mx.metric.logger$new()

#Run model
model <- mx.model.FeedForward.create(
  lro, 
  X=train.x,
  y=train.y,
  eval.data=list(data=test.x, label=test.y),
  array.layout="rowmajor",
  ctx=mx.cpu(), 
  num.round=10, 
  array.batch.size=NROW(train.x), #score hole train set at one time
  learning.rate=2e-6, 
  momentum=0.9, 
  eval.metric=mx.metric.mse,
  epoch.end.callback = mx.callback.save.checkpoint("boston"),
  batch.end.callback = mx.callback.log.train.metric(1, logger) 
)

#see the log:
testmse <- logger$eval
hist(testmse)
#trainmse <- logger$train

#get the index of the lowest mse
bestmodelno <- as.numeric(which(logger$eval == min(logger$eval)))

#load the best model for prediction:
Bestmodel <- mx.model.load("boston",bestmodelno)






