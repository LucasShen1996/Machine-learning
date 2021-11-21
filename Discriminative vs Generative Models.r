install.packages("mvtnorm")

# Load libraries
library(ggplot2)
library(reshape2)
library(mvtnorm) 

# Load data
train <- read.csv("./Data_set/Task1E_train.csv")
test <- read.csv("./Data_set/Task1E_test.csv")


train.data <- train[,-3] # grab all columns leave out the species (last column)
train.label <- train[,3]
test.data <- test[,-3] # grab all columns leave out the species (last column)
test.label <- test[,3]

# Bayesian classifier (BC)
BayesianClassifier <- function(train.data,train.label,test.data){
    
    # Class probabilities:
    p0.hat <- sum(train.label==1)/nrow(train.data) # total number of samples in class 0 divided by the total nmber of training data
    p1.hat <- sum(train.label==-1)/nrow(train.data) # or simply 1 - p1.hat

    # Class means:
    mu0.hat <- colMeans(train.data[train.label==1,])
    mu1.hat <- colMeans(train.data[train.label==-1,])
    
    # class covariance matrices:
    sigma0.hat <- var(train.data[train.label==1,])
    sigma1.hat <- var(train.data[train.label==-1,])
    
    # shared covariance matrix:
    sigma.hat <- p0.hat * sigma0.hat + p1.hat * sigma1.hat 
    
    # calculate posteriors:
    posterior0 <- p0.hat*dmvnorm(x=train.data, mean=mu0.hat, sigma=sigma.hat)
    posterior1 <- p1.hat*dmvnorm(x=train.data, mean=mu1.hat, sigma=sigma.hat)
    
    # Predict on testing data
    train.predict <- ifelse(p0.hat*dmvnorm(x=train.data, mean=mu0.hat, sigma=sigma.hat) > p1.hat*dmvnorm(x=train.data, mean=mu1.hat, sigma=sigma.hat), 1, -1)
    test.predict <- ifelse(p0.hat*dmvnorm(x=test.data, mean=mu0.hat, sigma=sigma.hat) > p1.hat*dmvnorm(x=test.data, mean=mu1.hat, sigma=sigma.hat), 1, -1)
    #missclasified error for train and test
    train.error <- sum((train.predict != train.label)/nrow(train.data))*100
    test.error <- sum((test.predict != test.label)/nrow(test.data))*100
    
    return(c(train.error,test.error))
    }

# Logistic Regression (LR)
# refers to tutorial code

LogisticRegresson <- function(train.data,train.label,test.data){
 
    # auxiliary function that predicts class labels
    predict <- function(w, X, c0, c1){
    sig <- sigmoid(w, X)
    return(ifelse(sig>0.5, c1,c0))
    }
    
    # auxiliary function that calculate a cost function
    cost <- function (w, X, T, c0){
    sig <- sigmoid(w, X)
    return(sum(ifelse(T==c0, 1-sig, sig)))
    }
    
    # Sigmoid function (=p(C1|X))
    sigmoid <- function(w, x){
    return(1.0/(1.0+exp(-w%*%t(cbind(1,x)))))    
    }
    
    
    # Initializations
    c0 <- 1; c1 <- -1
    tau.max <- nrow(train.data)# maximum number of iterations
    eta <- 0.01 # learning rate
    epsilon <- 0.01 # a threshold on the cost (to terminate the process)
    tau <- 1 # iteration counter
    terminate <- FALSE

    ## Just a few name/type conversion to make the rest of the code easy to follow
    X <- as.matrix(train.data) # rename just for conviniance
    T <- ifelse(train.label==c0,0,1) # rename just for conviniance

    W <- matrix(,nrow=tau.max, ncol=(ncol(X)+1)) # to be used to store the estimated coefficients
    W[1,] <- runif(ncol(W)) # initial weight (any better idea?)

    # project data using the sigmoid function (just for convenient)
    Y <- sigmoid(W[1,],X)

    costs <- data.frame('tau'=1:tau.max)  # to be used to trace the cost in each iteration
    costs[1, 'cost'] <- cost(W[1,],X,T, c0)
    
    
    while(!terminate){
    # check termination criteria:
    terminate <- tau >= tau.max | cost(W[tau,],X,T, c0)<=epsilon
    
    # shuffle data:
    train.index <- sample(1:nrow(train.data), nrow(train.data), replace = FALSE)
    X <- X[train.index,]
    T <- T[train.index]
    
    # for each datapoint:
    for (i in 1:nrow(train.data)){
        # check termination criteria:
        if (tau >= tau.max | cost(W[tau,],X,T, c0) <=epsilon) {terminate<-TRUE;break}
        
        Y <- sigmoid(W[tau,],X)
            
        # Update the weights
        W[(tau+1),] <- W[tau,] - eta * (Y[i]-T[i]) * cbind(1, t(X[i,]))
        
        # record the cost:
        costs[(tau+1), 'cost'] <- cost(W[tau,],X,T, c0)
        
        # update the counter:
        tau <- tau + 1
        
        # decrease learning rate:
        eta = eta * 0.999
        }
    }
    # Done!
    costs <- costs[1:tau, ] # remove the NaN tail of the vector (in case of early stopping)

    # the  final result is:
    w <- W[tau,]
    #Do predict
    lr.train <- predict(w, train.data, c0, c1)
    lr.test <- predict(w,test.data,c0,c1)
    # calculating missclassified train and test error:
    train.error <- sum(lr.train != train.label)/nrow(train.data) * 100
    test.error <- sum(lr.test != test.label)/nrow(test.data) * 100
    
     return (c(train.error,test.error))
    
    } 

# the error matirx
error <- matrix(,nrow=nrow(train)/5+1,ncol=3)

### main loop for calculating misclassified train and test error 
### of model trained on batches of size incremented by 5 each time 
batch = 1      # counter
train.len <- nrow(train)
# dataframes to store train and test errors of both the models
bc_error <- data.frame('Batch_Size'=seq(5,nrow(train.data),5))
lr_error <- data.frame('Batch_Size'=seq(5,nrow(train.data),5))

# iteration of 5 more data points every time
for (i in seq(5,train.len,5)){
    
    # train data of size i where is a multiple of 5
    train.data.batch = train.data[1:i,]
    train.label.batch = train.label[1:i]
    
    # function call to get train and test error of models trained on data of size i
    bc <- BayesianClassifier(train.data.batch,train.label.batch ,test.data)
    lr  <-  LogisticRegresson(train.data.batch,train.label.batch,test.data)
    
    # store train and test error of Bayesian Classifier in dataframe for bc
    bc_error[batch,'bc_train_error'] = bc [1]
    bc_error[batch,'bc_test_error'] = bc [2]

    # store train and test error of Logistic Regression in dataframe for lr
    lr_error[batch,'lr_train_error'] = lr[1]
    lr_error[batch,'lr_test_error'] = lr [2]    
    
    batch = batch + 1      # update counter
}

plot(x=bc_error[,1],y=bc_error[,2], type="l", col="blue", xlab="Batch Size", ylab="train error")
lines(x=lr_error[,1],y=lr_error[,2], type="l", lty=2, lwd=2, col="red")
legend("topright",inset=.05, lty=c(2, 1),  c("Logistic Regresson","Bayesian Classifier"), col=c("red", "blue"))

plot(x=bc_error[,1],y=bc_error[,3], type="l", col="blue", xlab="Batch Size", ylab="test error")
lines(x=lr_error[,1],y=lr_error[,3], type="l", lty=2, lwd=2, col="red")
legend("topright",inset=.05, lty=c(2, 1),  c("Logistic Regresson","Bayesian Classifier"), col=c("red", "blue"))
