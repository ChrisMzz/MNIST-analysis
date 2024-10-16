# set working directory if using RStudio as IDE
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# if you're using something else you can run something like this :
# setwd(getSrcDirectory(function(){})[1])


library(dslabs) # contains mnist dataset
library(flexmix)

mnist <- read_mnist()

# MNIST images are of size 28x28
s = 28*28


# I prefer differentiating the number of total images 
# from the number of selected images
n_images = length(mnist$train$images)/s
n = 1000

# generate n training images in an array of shape (n, 28x28)
train_images = array(rep(0,s*n), c(n,s))
for (i in 1:n) {
  train_images[i,] = matrix(t(mnist$train$images)[((i-1)*s+1):(i*s)], nrow=28)[,28:1]
}


####### PREPROCESSING #######

threshold_value = 127
train_images <- ifelse(train_images > threshold_value, 1, 0)

# check if preprocessing works as intended on one of the images
image(1:28, 1:28, matrix(train_images[5,], nrow=28),
      col = gray(seq(0, 1, 0.05)), xlab = "", ylab="")


#############################


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
layout(matrix(1:10, 2, 5, byrow = TRUE))
params = parameters(flexmix_model)
for (k in 1:10) {
  comp = matrix(params[,k], nrow=28)
  image(1:28, 1:28, comp, col = gray(seq(0, 1, 0.05)), xlab = "", ylab="")
}
# each cluster can be seen as an operator taking the weighted average of 
# the pixels of an input image, where the weights for each cluster
# are what's being displayed in the figures shown from the code above



# generate and display confusion matrix
tab_data = table(cluster_assignments, train_labels_sample)
print(tab_data)
# This allows us to check which numbers are difficult to classify, 
# which numbers are mistaken with others, etc.









