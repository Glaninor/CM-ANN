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
  layer_dense(units = 256, activation = 'relu', input_shape = c(1276)) %>%
  layer_dropout(rate = 0.1) %>% 
  layer_dense(units = 1, activation = 'sigmoid') %>% 
  compile( 
    optimizer = optimizer_rmsprop(),
    loss = loss_binary_crossentropy,
    metrics = c('accuracy')
  )

history <- model %>% fit(keras.train,keras.train.labels, epochs=30, batch_size=256)
model %>% evaluate(keras.test,keras.test.labels)
plot(history)

Epoch 1/30
2505/2505 [==============================] - 0s 108us/sample - loss: 740547.0594 - acc: 0.5138
Epoch 2/30
2505/2505 [==============================] - 0s 87us/sample - loss: 389626.6626 - acc: 0.5948
Epoch 3/30
2505/2505 [==============================] - 0s 78us/sample - loss: 260628.1891 - acc: 0.6475
Epoch 4/30
2505/2505 [==============================] - 0s 76us/sample - loss: 223590.1432 - acc: 0.6842
Epoch 5/30
2505/2505 [==============================] - 0s 83us/sample - loss: 148271.5754 - acc: 0.7477
Epoch 6/30
2505/2505 [==============================] - 0s 94us/sample - loss: 153276.3894 - acc: 0.7493
Epoch 7/30
2505/2505 [==============================] - 0s 94us/sample - loss: 117974.5497 - acc: 0.7768
Epoch 8/30
2505/2505 [==============================] - 0s 94us/sample - loss: 117897.7414 - acc: 0.7884
Epoch 9/30
2505/2505 [==============================] - 0s 77us/sample - loss: 97155.1186 - acc: 0.8048
Epoch 10/30
2505/2505 [==============================] - 0s 87us/sample - loss: 96574.8798 - acc: 0.8140
Epoch 11/30
2505/2505 [==============================] - 0s 94us/sample - loss: 92372.4791 - acc: 0.8140
Epoch 12/30
2505/2505 [==============================] - 0s 93us/sample - loss: 95592.5334 - acc: 0.8291
Epoch 13/30
2505/2505 [==============================] - 0s 76us/sample - loss: 77353.1247 - acc: 0.8451
Epoch 14/30
2505/2505 [==============================] - 0s 93us/sample - loss: 82436.7417 - acc: 0.8387
Epoch 15/30
2505/2505 [==============================] - 0s 92us/sample - loss: 99097.1579 - acc: 0.8311
Epoch 16/30
2505/2505 [==============================] - 0s 93us/sample - loss: 71941.7554 - acc: 0.8639
Epoch 17/30
2505/2505 [==============================] - 0s 93us/sample - loss: 77712.3819 - acc: 0.8523
Epoch 18/30
2505/2505 [==============================] - 0s 93us/sample - loss: 66454.0681 - acc: 0.8754
Epoch 19/30
2505/2505 [==============================] - 0s 96us/sample - loss: 81619.4163 - acc: 0.8555
Epoch 20/30
2505/2505 [==============================] - 0s 95us/sample - loss: 78217.4005 - acc: 0.8623
Epoch 21/30
2505/2505 [==============================] - 0s 89us/sample - loss: 80982.7516 - acc: 0.8663
Epoch 22/30
2505/2505 [==============================] - 0s 89us/sample - loss: 69378.6875 - acc: 0.8750
Epoch 23/30
2505/2505 [==============================] - 0s 92us/sample - loss: 81795.6718 - acc: 0.8611
Epoch 24/30
2505/2505 [==============================] - 0s 92us/sample - loss: 72887.1608 - acc: 0.8743
Epoch 25/30
2505/2505 [==============================] - 0s 86us/sample - loss: 69498.4728 - acc: 0.8679
Epoch 26/30
2505/2505 [==============================] - 0s 76us/sample - loss: 77013.6860 - acc: 0.8719
Epoch 27/30
2505/2505 [==============================] - 0s 92us/sample - loss: 69297.6532 - acc: 0.8786
Epoch 28/30
2505/2505 [==============================] - 0s 93us/sample - loss: 70583.0148 - acc: 0.8631
Epoch 29/30
2505/2505 [==============================] - 0s 95us/sample - loss: 63511.5331 - acc: 0.8743
Epoch 30/30
2505/2505 [==============================] - 0s 93us/sample - loss: 71529.8714 - acc: 0.8754

