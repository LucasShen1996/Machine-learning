# auxiliary function to calculate labels based on the estimated coefficients
predict_func <- function(Phi, w){
    return(Phi%*%w)
} 

# auxiliary function to calculate the objective function for the training
train_obj_func <- function (Phi, w, label, lambda){
    # the L2 regulariser is already included in the objective function for training 
    return(.5 * (mean((predict_func(Phi, w) - label)^2)) + .5 * lambda * w %*% w)
}

# auxiliary function to compute the error of the model
get_errors <- function(train_data, test_data, W) {
  n_weights = dim(W)[1]
  errors = matrix(,nrow=n_weights, ncol=2)
  for (tau in 1:n_weights) {
    errors[tau,1] = train_obj_func(train_data$x, W[tau,],train_data$y, 0)
    errors[tau,2] = train_obj_func(test_data$x, W[tau,],test_data$y, 0)
  }
  return(errors)
}

##--- Stochastic Gradient Descent --------------------------------------------
sgd_train <- function(train_x, train_y, lambda, eta, epsilon, max_epoch) {

   train_len = dim(train_x)[1]
   tau_max = max_epoch * train_len

   W <- matrix(,nrow=tau_max, ncol=ncol(train_x)) 
   W[1,] <- runif(ncol(train_x))
  
   tau = 1 # counter 
   obj_func_val <-matrix(,nrow=tau_max, ncol=1) 
   obj_func_val[tau,1] = train_obj_func(train_x, W[tau,],train_y, lambda)

   while (tau <= tau_max){

       # check termination criteria
       if (obj_func_val[tau,1]<=epsilon) {break}
 
       # shuffle data:
       train_index <- sample(1:train_len, train_len, replace = FALSE)
    
       # loop over each datapoint
       for (i in train_index) {
           # increment the counter
           tau <- tau + 1
           if (tau > tau_max) {break}

           # make the weight update
           y_pred <- predict_func(train_x[i,], W[tau-1,])
           W[tau,] <- sgd_update_weight(W[tau-1,], train_x[i,], train_y[i], y_pred, lambda, eta)

           # keep track of the objective funtion
           obj_func_val[tau,1] = train_obj_func(train_x, W[tau,],train_y, lambda)
       }
   }
   # resulting values for the training objective function as well as the weights
   return(list('vals'=obj_func_val,'W'=W))
}


# updating the weight vector
sgd_update_weight <- function(W_prev, x, y_true, y_pred, lambda, eta) {
   # MODIFY THIS FUNCTION FOr L2 REG
   grad = - (y_true-as.vector(y_pred)) * x 
   return(W_prev - eta * (grad+lambda*W_prev))
}

# reading the data
read_data <- function(fname, sc) {
   data <- read.csv(file=fname,head=TRUE,sep=",")
   nr = dim(data)[1]
   nc = dim(data)[2]
   x = data[1:nr,1:(nc-1)]
   y = data[1:nr,nc]
   if (isTRUE(sc)) {
      x = scale(x)
      y = scale(y)
   }
   return (list("x" = x, "y" = y))
}

train = read_data("./Data_set/Task1C_train.csv", TRUE)
test = read_data("./Data_set/Task1C_test.csv", TRUE)

#parameter settings 
max_epoch = 20
epsilon = .001
eta = .01

library(ggplot2)

lambda =seq(0,10,by = 0.4)
lambda_error =  matrix(,nrow=length(lambda), ncol=3)

for (i in 1:length(lambda) ){
    train_res = sgd_train(train$x, train$y, lambda[i], eta, epsilon, max_epoch)
    errors = get_errors(train, test, train_res$W)
    lambda_error[i,1] = lambda[i]
    lambda_error[i,2] =mean(errors[,1])
    lambda_error[i,3] =mean(errors[,2])
}

plot(x=log(lambda_error[,1]),y=lambda_error[,2], type="l", col="blue",ylim=c(0.3,0.55), xlab="log(lambda)", ylab="error")
lines(x=log(lambda_error[,1]),y=lambda_error[,3], type="l", lty=2, lwd=2, col="red")
legend("topright",inset=.05, lty=c(2, 1),  c("test","train"), col=c("red", "blue"))

which(lambda_error==min(lambda_error[,3]),arr.ind=TRUE)

lambda_error[7,]
