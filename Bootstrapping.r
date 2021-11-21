# define a function that generates sample indixes based on bootstrap technique
boot <- function (original.size=100, sample.size=original.size, times=100){
    indx <- matrix(nrow=times, ncol=sample.size)
    for (t in 1:times){
        indx[t, ] <- sample(x=original.size, size=sample.size, replace = TRUE)
    }
    return(indx)
}

# Import library
library(ggplot2)
library(reshape2)
library(corrplot)


# Load data
train <- read.csv('./Data_set/Task1B_train.csv')
test <- read.csv('./Data_set/Task1B_test.csv')
train_data = train[-5]
train_label = train[5]
test_data = test[-5]
test_label = test[5]

# KNN function (distance should be one of euclidean, maximum, manhattan, canberra, binary or minkowski)
knn <- function(train.data, train.label, test.data, K=3, distance = 'euclidean'){
    ## count number of train samples
    train.len <- nrow(train.data)    
    ## count number of test samples
    test.len <- nrow(test.data)    
    ## calculate distances between samples
    dist <- as.matrix(dist(rbind(test.data, train.data), method= distance))[1:test.len, (test.len+1):(test.len+train.len)]
    test.label={}
    ## for each test sample...
    for (i in 1:test.len){
        ### ...find its K nearest neighbours from training sampels...
        nn <- as.data.frame(sort(dist[i,], index.return = TRUE))[1:K,2]      
        ###... and calculate the predicted labels according to the majority vote
        test.label[i]<- (mean(train.label[nn]))
    }    
    ## return the class labels as output
    return (test.label)
}

K <- 15         
L <- 50          
N <- 60 

boot.indx <- boot(nrow(train_data),N,L)

mse <- data.frame('K'=1:K, 'L'=1:L, 'test'=rep(0,L*K))
for (k in 1: K){
    ### for every dataset sizes:
    for (l in 1:L){
        #### calculate iteration index i
        i <- (k-1)*L+l
        #### save sample indices that were selected by bootstrap
        indx <- boot.indx[l,]
        #### save the value of k and l
        mse[i,'K'] <- k
        mse[i,'L'] <- l
        #### calculate and record the train and test missclassification rates
        mse[i,'test'] <-  sum(test_label - knn(train_data[indx, ], train_label[indx,], test_data, K=k)^2)/nrow(test_data)
    } 
}

# boxplot
mse.m <- melt(mse, id=c('K', 'L')) # reshape for visualization
names(mse.m) <- c('K', 'L', 'type', 'MSE')
ggplot(data=mse.m[mse.m$type=='test',], aes(factor(K), MSE,fill=type)) + geom_boxplot(outlier.shape = NA)  + 
  scale_color_discrete(guide = guide_legend(title = NULL)) + 
  ggtitle('Mean Squared Error vs. K (Box Plot)') + theme_minimal()
# ignore the warnings (because of ignoring outliers)
options(warn=-1)

# set parameters
K <- 10                     # K for KNN 
L <- seq(10,200,10)         # number of datasets
N <- 30                     # size of datasets


mse <- data.frame('times'=0, 'L'=0, 'test'=0)
for (times in L){
    mse <- rbind(mse,data.frame('times'=times,'L'=1:times,'test'=rep(0,times)))
}
for (l in seq(10,200,10)){
# main loop

    boot.indx <- boot(nrow(train_data),N,l)
        ### for every dataset sizes:
        for (a in 1:l){
            #### calculate iteration index i
            i <- 5*(l/10)*(l/10-1)+a
            #### save sample indices that were selected by bootstrap
            indx <- boot.indx[a,]
            #### save the value of k and l
            mse[i,'L'] <- a
            mse[i,'times'] <- l
            #### calculate and record the train and test missclassification rates
            mse[i,'test'] <-  mean(sum((test_label - knn(train_data[indx, ], train_label[indx,], test_data, K=10))^2)/nrow(test_data))
        } 
}

# boxplot
mse.m <- melt(mse, id=c('times', 'L')) # reshape for visualization
names(mse.m) <- c('times', 'L', 'type', 'AvgError')
ggplot(data=mse.m[mse.m$type=='test',], aes(factor(times), AvgError,fill=type)) + geom_boxplot(outlier.shape = NA)  + 
  scale_color_discrete(guide = guide_legend(title = NULL)) + 
  ggtitle('Average testing Error vs. Times of Bootstrapping (Box Plot)') + theme_minimal()


