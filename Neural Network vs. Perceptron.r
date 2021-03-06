library(ggplot2)
library(reshape2)

# reading the train and test data
train = read.csv("./Dataset and Sample code/Task2B_train.csv")
test =  read.csv("./Dataset and Sample code/Task2B_test.csv")

# plot if the training data
ggplot(data=train, aes(x=x1, y=x2, colour=factor(y))) + 
        geom_point()

list <- structure(NA,class="result")
"[<-.result" <- function(x,...,value) {
  args <- as.list(match.call())
  args <- args[-c(1:2,length(args))]
  length(value) <- length(args)
  for(i in seq(along=args)) {
    a <- args[[i]]
    if(!missing(a)) eval.parent(substitute(a <- v,list(a=a,v=value[[i]])))
  }
  x
}

# reading the data
read.data <- function(file.name, scaling=FALSE) {
  data <- read.csv(file=file.name,head=TRUE,sep=",")
  data <- data[complete.cases(data),] # removes rows with NA values
  D <- ncol(data)
  x = data[,-D]
  y = data[,D]
  if (isTRUE(scaling)) {
    x = scale(x)
    y = scale(y)
  }
  return (list('x' = x, 'y' = y))
}

error.rate <- function(Y1, T1){
  if (length(Y1)!=length(T1)){
    stop('error.rate: size of true lables and predicted labels mismatch')
  }
  return (sum(T1!=Y1)/length(T1))
}

####################### PERCEPTRON #######################
## prediction
perceptron.predict <- function(Phi, W){
  return(ifelse(Phi%*%W>=0, +1, -1))
}
## is it a misclassification? if yes, update the weight vector
is.a.miss <- function(Phi, W, T1){
  return((W%*%Phi)*T1<0)
}
## Perceptron Build function
perceptron.build <- function(X1, T1, eta=0.01, epsilon=0.001, tau.max=100, plotting=FALSE){
  if (length(unique(T1))!=2){
    stop("Perceptron: The input data is not a binary classification problem.")
  }
  if (all(sort(unique(T1)) != c(-1, 1))) {
    stop("Perceptron: The labels are not (-1, +1).")
  }
  
  N1 <- nrow(X1)
  Phi <- as.matrix(cbind(1, X1)) # add a column of 1 as phi_0

  W <- matrix(NA,nrow=tau.max, ncol=ncol(Phi)) # Empty Weight vector
  W[1,] <- 2*runif(ncol(Phi))-1 # Random initial values for weight vector
  error.rec <- matrix(NA,nrow=tau.max, ncol=1) # Placeholder for errors
  error.rec[1] <- error.rate(perceptron.predict(Phi, W[1,]), T1) # record error for initial weights
  tau <- 1 # iteration counter 
  terminate <- FALSE # termination status
  
  while(!terminate){
    # resuffling train data and associated labels:
    indx <- sample(1:N1, replace = FALSE)
    Phi <- Phi[indx,]
    T1 <- T1[indx]
    
    for (i in 1:N1){
      if (tau >= tau.max) {break}
      # look for missclassified samples
      if (is.a.miss(Phi[i,], W[tau,], T1[i])){
        tau <- tau +1                                 # update tau counter
        W[tau,] <- W[tau-1,] + eta * Phi[i,] * T1[i]  # update the weights
        error.rec[tau] <- error.rate(perceptron.predict(Phi, W[tau,]), T1)# update the records
        eta = eta * 0.99                                 # decrease eta
      } 
    }
    
    # recalculate termination conditions
    terminate <- tau >= tau.max | (abs(error.rec[tau] - error.rec[tau-1]) <= epsilon )
    
  }
  if (plotting){
    plot(error.rec[complete.cases(error.rec),], xlab = 'tau', ylab = 'error', main = 'Perceptron')
  }
  W <- W[complete.cases(W),]  # cut the empty part of the matrix (when the loop stops before tau == tau.max)
  return(W[nrow(W),])         # return the last wight vector
}

# Read the datasets
set.seed(1234)          # set random seed
library(ggplot2)        # load libraries
list[X1,T1] <- read.data('./Dataset and Sample code/Task2B_train.csv') # read training data
T1[T1==0] <- -1         # convert 0 labels to -1 
list[X2,T2] <- read.data('./Dataset and Sample code/Task2B_test.csv') # read test data
T2[T2==0] <- -1         # convert 0 labels to -1 

W_001<-perceptron.build(X1, T1,eta=0.01, tau.max = 1000, plotting = TRUE)

W_009<-perceptron.build(X1, T1,eta=0.09, tau.max = 1000, plotting = TRUE)

# Evaluate Perceptron (TO BE COMPLETE)
## Hint: compute Phi, predict the test labels based on the model from the above statements, and then compare the predicted labels with the real labels
Phi1 <- as.matrix(cbind(1,X2))
predict.result_001 <- perceptron.predict(Phi1,W_001)
test.result_001 <- as.matrix(cbind(X2,predict.result_001))
test.result_001 <- as.data.frame(test.result_001)
names(test.result_001) <- c('x1','x2','y')
# Calculate test error
error_001 <- error.rate(predict.result_001,T2)
print(paste('error rate is :' ,error_001))
# Plot test result
ggplot(data=test.result_001, aes(x=x1,y=x2,color=factor(y))) + geom_point()

# Evaluate Perceptron (TO BE COMPLETE)
## Hint: compute Phi, predict the test labels based on the model from the above statements, and then compare the predicted labels with the real labels
Phi9 <- as.matrix(cbind(1,X2))
predict.result_009 <- perceptron.predict(Phi9,W_009)
test.result_009 <- as.matrix(cbind(X2,predict.result_009))
test.result_009 <- as.data.frame(test.result_009)
names(test.result_009) <- c('x1','x2','y')
# Calculate test error
error_009 <- error.rate(predict.result_009,T2)
print(paste('error rate is :' ,error_009))
# Plot test result
ggplot(data=test.result_009, aes(x=x1,y=x2,color=factor(y))) + geom_point()

## the activation function (tanh here)
h <- function(z) { 
  return ((exp(z)-exp(-z))/(exp(z)+exp(-z)))
}

## the derivitive of the activation function (tanh here)
h.d <- function(z) {
return (1-(h(z))^2)
}

## Class Probabilities
class.prob <- function(X, W1, W2, b1, b2){
  a2 <- h(sweep(W1 %*% X, 1, b1,'+' ))
  a3 <- h(sweep(W2 %*% a2, 1, b2,'+' ))
  return (a3)
}

## prediction
nn.predict <- function(X, W1, W2, b1, b2, threshold=0){
  return (ifelse(class.prob(X, W1, W2, b1, b2)>=threshold, 1, -1))
}

## feedforward step
feedforward <- function(Xi, Ti, W1, b1, W2, b2){
  ### 1st (input) layer 
  a1 <- Xi
  y <- Ti
  ### 2nd (hidden) layer
  z2 <- W1 %*% a1 + b1
  a2 <- h(z2)        
  ### 3rd (output) layer
  z3 <- W2 %*% a2 + b2
  a3 <- h(z3)  
  return(list(a1, a2, a3, y, z2, z3))
}

## backpropagation step
backpropagation <- function(Ti, W2, z2, z3, a3){
  ### 3rd (output) layer
  d3 <- -(Ti-a3) * h.d(z3)
  ### 2nd (hidden) layer
  d2 <-  t(W2)%*%d3  * h.d (z2)
  return(list(d2,d3))
}

## NN build function
nn.build <- function(K, X1, T1, plotting=FALSE, epoch.max=50, eta = 0.1, lambda = 0.01){
  # initialization
  if (plotting) {error.rec <- matrix(NA,nrow=epoch.max, ncol=1)}
  D <- nrow(X1)
  if (D!=2) {stop('nn.predict: This simple version only accepts two dimensional data.')}
  N <- ncol(X1)

  W1 <- matrix(rnorm(D*K, sd=0.5), nrow=K, ncol=D)
  b1 <- matrix(rnorm(1*K), nrow=K, ncol=1)
  W2 <- matrix(rnorm(K*1, sd=0.5), nrow=1, ncol=K)
  b2 <- matrix(rnorm(1*1), nrow=1, ncol=1)

  for (epoch in 1:epoch.max){   
    ## delta vectors/matrices initialization
    W1.d <- W1 *0
    b1.d <- b1 *0
    W2.d <- W2 *0
    b2.d <- b2 *0

    for (i in 1:N){
      ## Feedforward:
      list[a1, a2, a3, y, z2, z3] <- feedforward(X1[,i], T1[i], W1, b1, W2, b2)          
      ## Backpropagation:
      list[d2, d3] <- backpropagation(T1[i], W2, z2, z3, a3)
      ## calculate the delta values
      ### 1st layer
      W1.d <- W1.d + d2 %*% t(a1)
      b1.d <- b1.d + d2
      ### 2nd layer
      W2.d <- W2.d + d3 %*% t(a2)
      b2.d <- b2.d + d3
    }
    ## update weight vectors and matrices
    W1 <- W1 - eta * (W1.d/N + lambda*W1)
    b1 <- b1 - eta * (b1.d/N)
    W2 <- W2 - eta * (W2.d/N + lambda*W2)
    b2 <- b2 - eta * (b2.d/N)
    ## record the errors
    if (plotting){error.rec[epoch]<- error.rate(nn.predict(X1, W1, W2, b1, b2), T1)}
  }
  if (plotting){plot(error.rec, xlab = 'epoch', ylab = 'error', main = 'Neural Net')}
  return(list(W1, W2, b1, b2))
}

X1.t <- t(as.matrix(X1))
X2.t <- t(as.matrix(X2))

K <- seq(5,100,5)  #different numbers of units in hidden layer
errors <- data.frame('K'=K)   # dataframe to store error every time
for (k in K) {
  list[W1_001, W2_001, b1_001, b2_001]<- nn.build(k, X1.t, T1, plotting=FALSE, epoch.max=50, eta = 0.01, lambda = 0.01)
  list[W1_009, W2_009, b1_009, b2_009]<- nn.build(k, X1.t, T1, plotting=FALSE, epoch.max=50, eta = 0.09, lambda = 0.01)  
  #Record the test errors for plotting purposes (TO BE COMPLETE) 
  errors[k/5,"Eta_0.01"] <- error.rate(nn.predict(X2.t, W1_001, W2_001, b1_001, b2_001), T2)  
  errors[k/5,"Eta_0.09"] <- error.rate(nn.predict(X2.t, W1_009, W2_009, b1_009, b2_009), T2)  
}

# plot misclassification error of both models for test data set 
ggplot(data=errors) + geom_line(aes(x=K, y=Eta_0.01, color='Eta 0.01')) + geom_line(aes(x=K, y=Eta_0.09, color='Eta 0.09')) +
       scale_color_discrete(guide = guide_legend(title = NULL)) + theme_minimal() +
       ggtitle("TESTING ERROR V/S NO. OF HIDDEN UNITS")

k.best <- 40

list[W1_009, W2_009, b1_009, b2_009]<- nn.build(k.best, X1.t, T1, plotting=FALSE, epoch.max=50, eta = 0.09, lambda = 0.01)

predict.label <- nn.predict(X2.t, W1_009, W2_009, b1_009, b2_009)  

predict.result <- as.data.frame(cbind(X2,T2,t(predict.label))) 
names(predict.result) <- c('x1','x2','y')

ggplot(data = predict.result,aes(x=x1,y=x2))+geom_point(aes(color=factor(y)))+scale_shape_identity()
