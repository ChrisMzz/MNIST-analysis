# set working directory if using RStudio as IDE
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

options(timeout = max(10000, getOption("timeout")))

remove.packages("keras")


library(keras3)
#reticulate::install_python(version = '3.11')
#install_keras(tensorflow = "gpu")
#install_tensorflow()

mnist = dataset_mnist()


x_train = mnist$train$x
y_train = mnist$train$y
x_test = mnist$test$x
y_test = mnist$test$y


batch_size = 128
num_classes = 10
epochs = 50

# Input image dimensions
img_rows = 28
img_cols = 28



x_train = array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test = array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))

x_train_shifted = array_reshape(0*(1:(nrow(x_train)*56^2)),c(nrow(x_train),56,56,1))
x_test_shifted = array_reshape(0*(1:(nrow(x_test)*56^2)),c(nrow(x_test),56,56,1))

for (k in 1:nrow(x_train)) {
  train_k = floor(runif(n=2,min=1,max=27))
  x_train_shifted[k,train_k[1]:(train_k[1]+27),train_k[2]:(train_k[2]+27),1] = x_train[k,1:28,1:28,1]
}

for (k in 1:nrow(x_test)) {
  test_k = floor(runif(n=2,min=1,max=27))
  x_test_shifted[k,test_k[1]:(test_k[1]+27),test_k[2]:(test_k[2]+27),1] = x_test[k,1:28,1:28,1]
}

img_rows = 56
img_cols = 56



input_shape = c(img_rows, img_cols, 1)

x_train_shifted = x_train_shifted / 255
x_test_shifted = x_test_shifted / 255


y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

cnn_model = keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = input_shape) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')

cnn_model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

cnn_history = cnn_model %>% fit(
  x_train_shifted, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
)

cnn_model %>% evaluate(x_test_shifted, y_test)

# model prediction
cnn_pred <- cnn_model %>% 
  predict(x_test_shifted) %>% op_argmax()
head(cnn_pred, n=50)

cnn_pred_v = as.vector(cnn_pred)

## number of mis-classcified images
missed = as.vector(cnn_pred != mnist$test$y)
sum(missed)
# gets every image wrong, so it doesn't work and I can't figure out why



missed_image = mnist$test$x[missed,,]
missed_digit = mnist$test$y[missed]
missed_pred = cnn_pred_v[missed]


layout(matrix(1:9, 3, 3, byrow = TRUE))
for (index_image in 1:9) {
  input_matrix = missed_image[index_image,1:56,1:56]
  output_matrix = apply(input_matrix, 2, rev)
  output_matrix = t(output_matrix)
  image(1:56, 1:56, output_matrix, col=gray.colors(256), xlab=paste('Label ', missed_digit[index_image], ', Pred ', missed_pred[index_image]), ylab="")
}

# seems to get 0% of correct values


# next step : display confusion matrix for this version
# check if 4/7/9 and 3/5/8 are confused like in the mixed model

