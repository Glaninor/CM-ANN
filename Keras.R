library(devtools)
devtools::install_github("rstudio/keras")
library(keras)
library(tensorflow)
install_tensorflow()

ntrain = sample(nrow(all.judge),floor(0.7*nrow(all.judge)),replace=FALSE)
train = all.judge[ntrain,]
test = all.judge[-ntrain,]
train[which(is.na(train[1,]))]
train <- train[,-which(names(train)=="mcq240y")]
train <- train[,-which(names(train)=="mcq240r")]
train <- train[,-which(names(train)=="mcq240k")]
train <- train[,-which(names(train)=="mcq240i")]
train <- train[,-which(names(train)=="mcq240d")]
train <- train[,-which(names(train)=="mcq230d")]
test <- test[,-which(names(test)=="mcq240y")]
test <- test[,-which(names(test)=="mcq240r")]
test <- test[,-which(names(test)=="mcq240k")]
test <- test[,-which(names(test)=="mcq240i")]
test <- test[,-which(names(test)=="mcq240d")]
test <- test[,-which(names(test)=="mcq230d")]

keras.train <- as.matrix(train[,3:1278])
keras.train.labels <- as.matrix(train[,2])
keras.test <- as.matrix(test[,3:1278])
keras.test.labels <- as.matrix(test[,2])

model <- keras_model_sequential() 

model %>% 
  layer_dense(units = 64, activation = 'relu', input_shape = c(1276)) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid') %>% 
  compile( 
    optimizer = optimizer_rmsprop(),
    loss = loss_binary_crossentropy,
    metrics = c('accuracy')
  )

history <- model %>% fit(keras.train,keras.train.labels, epochs=30, batch_size=64)
model %>% evaluate(keras.test,keras.test.labels)
plot(history)

