library(devtools)
devtools::install_github("rstudio/keras")
library(keras)
library(tensorflow)
install_tensorflow()


keras.train <- as.matrix(train[,3:20])
keras.train.labels <- as.matrix(train[,2])
keras.test <- as.matrix(test[,3:20])
keras.test.labels <- as.matrix(test[,2])

model <- keras_model_sequential() 

model %>% 
  layer_dense(units = 32, activation = 'relu', input_shape = c(18)) %>% 
  layer_dense(units = 1, activation = 'sigmoid') %>% 
  compile( 
    optimizer = optimizer_rmsprop(),
    loss = loss_binary_crossentropy,
    metrics = metric_binary_accuracy
  )

model %>% fit(keras.train,keras.train.labels, epochs=20, batch_size=32)
score <- model %>% evaluate(keras.test,keras.test.labels, batch_size =32)

score <- model %>% evaluate(keras.test,keras.test.labels, batch_size =32)
1074/1074 [==============================] - 0s 90us/sample - loss: 5032.7191 - binary_accuracy: 0.6006
