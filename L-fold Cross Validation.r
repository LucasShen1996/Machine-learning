library(reshape2)
library(ggplot2)
library(corrplot)

q2_train <- read.csv('./Data_set/Task1A_train.csv')

# define an auxiliary function that calculates the majority votes (or mode!)
majority <- function(x) {
   uniqx <- unique(x)
   uniqx[which.max(tabulate(match(x, uniqx)))]
}

# KNN function (distance should be one of euclidean, maximum, manhattan, canberra, binary or minkowski)
knn <- function(train.data, train.label, test.data, K=3){
    ## count number of train samples
    train.len <- nrow(train.data)    
    ## count number of test samples
    test.len <- nrow(test.data)    
    ## calculate distances between samples
    dist <- as.matrix(dist(rbind(test.data, train.data), method= 'euclidean'))[1:test.len, (test.len+1):(test.len+train.len)]
    test.label={}
    ## for each test sample...
    for (i in 1:test.len){
        ### ...find its K nearest neighbours from training sampels...
        nn <- as.data.frame(sort(dist[i,], index.return = TRUE))[1:K,2]      
        ###... and calculate the predicted labels 
        test.label[i]<- mean(train.label[nn,])
    }    
    ## return the class labels as output
    return (as.matrix(test.label))
}


# L-fold Cross Validation
cv <- function(train.data, train.label, numFold=10, K=3) {
    
  # Create L equal-size subsets
  folds <- cut(seq(1,nrow(train.data)),breaks=numFold,labels=FALSE)
  
  # Create a vector for MSE of L-fold 
  mse.lf <- rep(0,numFold)

  for(i in 1:numFold){
      
    #Divide the data by folds
    test_index <- which(folds==i,arr.ind=TRUE)
    test_data <- train.data[test_index,,drop=FALSE ]
    test_lable <- train.label[test_index,,drop=FALSE ]
    train_data <- train.data[-test_index,,drop=FALSE  ]
    train_label <- train.label[-test_index,,drop=FALSE ]
    mse <- sum((knn(train_data,train_label, test_data,K=K)-test_lable)^2)/nrow(test_data)
    mse.lf[i] <- mse
  }
  return(mse.lf)
}



q2_train_data <- q2_train[1]
q2_train_label <- q2_train[2]

# Use temp dataframe to store average of MSE of each K 
mse_cv <- as.matrix(0,nrow=15,ncol=1)
for (k in 1:15){
  mse_cv[k] <- mean(cv(q2_train_data, q2_train_label,K=k))   # mean of error
}


mse_cv_m<-cbind(seq(1,15,1),mse_cv)

## Plot 
plot(x=1/mse_cv_m[,1],y=mse_cv_m[,2], type="l", col="blue", xlab="1/K", ylab="average error")

