library(reshape2)
library(ggplot2)
library(corrplot)

train = read.csv("./Data_set/Task1A_train.csv")
test = read.csv("./Data_set/Task1A_test.csv")

train_data = train[1]
train_label = train[2]
test_data = test[1]
test_label = test[2]

# define an auxiliary function that calculates the majority votes (or mode!)
majority <- function(x) {
   uniqx <- unique(x)
   uniqx[which.max(tabulate(match(x, uniqx)))]
}

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
        test.label[i]<- mean(train.label[nn,])
    }    
    ## return the class labels as output
    return (test.label)
}

knn(train_data,train_label,test_data, K=4)

mse <- data.frame('K'=1:25, 'train'=rep(0,25), 'test'=rep(0,25))
for (k in 1:25){
    mse[k,'train'] <- sum((train_label-knn(train_data, train_label, train_data, K=k))**2)/nrow(train_label)
    mse[k,'test'] <-  sum((test_label-knn(train_data, train_label, test_data, K=k))**2)/nrow(test_label)
}

mse.m <- melt(mse, id='K') # reshape for visualization
names(mse.m) <- c('K', 'type', 'Mean_Squared_Error')
ggplot(data=mse.m, aes(x=log(1/K), y=Mean_Squared_Error, color=type)) + geom_line() +
       scale_color_discrete(guide = guide_legend(title = NULL)) + theme_minimal() +
       ggtitle("Mean Squared Error")

mse.m
