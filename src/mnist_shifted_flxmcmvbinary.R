# set working directory if using RStudio as IDE
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# if you're using something else you can run something like this :
# setwd(getSrcDirectory(function(){})[1])


library(dslabs) # contains mnist dataset
library(flexmix)



# predict_digit allows classification of the given image into 
# one of the model's component classes. Not designed for vectorization.
predict_digit = function (model, image) {
  params = parameters(model)
  m = 0
  K = 0
  for (k in 1:length(levels(factor(cluster_assignments)))) {
    accuracy = sum(image*params[,k])
    if (accuracy > m) {
      K = k
      m = accuracy
    }
  }
  if (m==0) { print("Image cannot be null! Prediction was 0.")}
  return(K-1)
}




mnist <- read_mnist()

# MNIST images are of size 28x28
s = 28*28
shift_s = s*4


# I prefer differentiating the number of total images 
# from the number of selected images
n_images = length(mnist$train$images)/s
n = 1000

# generate n training images in an array of shape (n, 28x28)
train_images = array(rep(0,shift_s*n), c(n,shift_s))
test_images = array(rep(0,shift_s*n), c(n,shift_s))
for (i in 1:n) {
  train_shift = matrix(rep(0,56^2), nrow=56)
  test_shift = matrix(rep(0,56^2), nrow=56)
  
  train_k = floor(runif(n=2,min=1,max=27))
  test_k = floor(runif(n=2,min=1,max=27))
  
  train_shift[train_k[1]:(train_k[1]+27),train_k[2]:(train_k[2]+27)] = matrix(t(mnist$train$images)[((i-1)*s+1):(i*s)], nrow=28)
  test_shift[test_k[1]:(test_k[1]+27),test_k[2]:(test_k[2]+27)] = matrix(t(mnist$test$images)[((i-1)*s+1):(i*s)], nrow=28)
  
  train_images[i,] = train_shift[,56:1]
  test_images[i,] = test_shift[,56:1]
}


####### PREPROCESSING #######

threshold_value = 127
train_images <- ifelse(train_images > threshold_value, 1, 0)
test_images <- ifelse(test_images > threshold_value, 1, 0)

# check if preprocessing works as intended on one of the images
image(1:56, 1:56, matrix(train_images[5,], nrow=56),
      col = gray(seq(0, 1, 0.05)), xlab = "", ylab="", xaxt="n", yaxt="n")


#############################




# Re-run code from here (without set seed) for further testing

set.seed(127)

# sample a small amount of data from the training set, 
# along with the corresponding labels
sample_idx <- sample(1:n, 800)
train_images_sample <- train_images[sample_idx, ]
train_labels_sample <- mnist$train$labels[sample_idx]


# train a model that will classify the images in 10 clusters
n_clusters <- 10
flexmix_model <- flexmix(train_images_sample ~ 1,
                         k = n_clusters,
                         model = FLXMCmvbinary())

# make sure the model classified into 10 classes (1 for each digit)
cluster_assignments <- clusters(flexmix_model)-1
if (length(levels(factor(cluster_assignments)))==10) {
  print("10 clusters") } else { print("Not 10 clusters") }

# show some generic information about the model
print(flexmix_model)


# generate figures for each cluster's parameter estimation
layout(matrix(1:6, 2, 3, byrow = TRUE))
params = parameters(flexmix_model)
for (k in 1:6) {
  comp = matrix(params[,k], nrow=56)
  image(1:56, 1:56, comp, col = gray(seq(0, 1, 0.05)), xlab = "", ylab="", xaxt="n", yaxt="n", main=paste("Class",k))
}
# each cluster can be seen as an operator taking the weighted average of 
# the pixels of an input image, where the weights for each cluster
# are what's being displayed in the figures shown from the code above



# generate and display confusion matrix
tab_data = table(cluster_assignments, train_labels_sample)
print(tab_data)
# This allows us to check which numbers are difficult to classify, 
# which numbers are mistaken with others, etc.



### VALIDATION


# resample new indices for test images
sample_idx = sample(1:n, 800)
test_images_sample = test_images[sample_idx, ]
test_labels_sample = mnist$test$labels[sample_idx]

# to test with the training sample
# test_images_sample = train_images_sample
# test_labels_sample = train_labels_sample


# make predictions
predictions = rep(0, length(sample_idx))
for (idx in 1:length(sample_idx)) {
  predictions[idx] = predict_digit(flexmix_model, test_images_sample[idx,])
}

# generate confusion matrix
pred_data = table(predictions, test_labels_sample)
print(pred_data)

# display both confusion matrices side by side
layout(matrix(1:2, ncol=2))
tab_dim = dim(tab_data)
pred_dim = dim(pred_data)
image(1:tab_dim[1], 1:tab_dim[2], t(matrix(tab_data, nrow =10))[,10:1],
      col = gray(seq(0, 1, 0.05)), xlab = "labels", ylab="predictions", main="Training Data Confusion Matrix")
image(1:pred_dim[1], 1:pred_dim[2], t(matrix(pred_data, nrow =10))[,10:1],
      col = gray(seq(0, 1, 0.05)), xlab = "labels", ylab="predictions", main="Test Data Confusion Matrix")



