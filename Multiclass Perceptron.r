train <- read.csv('./Data_set/Task1D_train.csv')
test <- read.csv('./Data_set/Task1D_test.csv')

#shuffle data
train.index <- sample(1:nrow(train), nrow(train), replace = FALSE)
test.index <- sample(1:nrow(test), nrow(test), replace = FALSE)
train = train[train.index,]
test= test[test.index,]

#change y column to factor
train$y = as.factor(train$y)
test$y = as.factor(test$y)

## Split data to train and test datasets
train.data <- train[, c('x1', 'x2','x3', 'x4')]
test.data <- test[, c('x1', 'x2','x3', 'x4')]
train.label <- train[, 'y']
test.label <- test[, 'y']

multiclass_perceptron <- function(train.data, train.label,test.data,test.label,eta) {
    train.len = nrow(train.data)
    test.len = nrow(test.data)
    # Stoping criterion
    epsilon <- 0.001
    # Maximum number of iterations
    tau.max <- nrow(train.data) /5    
    # iteration counter 
    tau <- 1     
    # termination status
    terminate <- FALSE     
    ## Basis function (Step 1)
    # add a column of 1 as phi_0
    Phi <- as.matrix(cbind(1, train.data))
    test.data <- as.matrix(cbind(1, test.data))
    # Empty Weight vector
    W <- matrix(,nrow=nlevels(train$y), ncol=ncol(Phi))
    # Random initial values for weight vector
    W[1,] <- runif(ncol(Phi)) 
    W[2,] <- runif(ncol(Phi))
    W[3,] <- runif(ncol(Phi))    
    # Placeholder for errors
    error <- matrix(0,nrow=tau.max/5, ncol=1)            
    # Main Loop (Step 2):
    while(!terminate){
        # iteration for batch of 5 training points
        for (i in seq(5,train.len,5)){
            if (tau > tau.max) {break}
                # for every point in batch of size 5
                for(j in (i-4):i){
                    # predict training point
                    y_pre<-c(Phi[j,]%*%W[1,],Phi[j,]%*%W[2,],Phi[j,]%*%W[3,])
                    y <- which.max(y_pre)
                    label = paste(c("C", y), collapse = "")              
                    # look for missclassification
                    if(train.label[j] != label){
                        # Update wrong label
                        W[y,] <- W[y,] - eta * Phi[j,]
                        #Update true label 
                        W[eval(parse(text=train.label[j])),] <- W[eval(parse(text=train.label[j])),] + eta * Phi[j,]    
                    }
                }        
            error_count = 0           
            #using test data count error
            for(k in 1:test.len){
              # predict test point
              y_pre<-c(test.data[k,]%*%W[1,],test.data[k,]%*%W[2,],test.data[k,]%*%W[3,])
              y <- which.max(y_pre)
              label = paste(c("C", y), collapse = "")
              # check prediction
              if (label!=test.label[k]){
                #count error
                error_count = error_count + 1          
              }
            }            
            # update the error
            error[tau] <- (error_count/test.len)*100
            # update tau counter
            tau <- tau + 1         
          }
        # recalculate termination conditions
        terminate <- tau >= tau.max | abs((error[tau]) - (error[tau-1])) <= epsilon   
    }
    ## report
    cat('\n\nThe  final weight vector:', W[1,])
    cat('\n\nThe  final weight vector:', W[2,])
    cat('\n\nThe  final weight vector:', W[3,])
    errors<-cbind(seq(5,train.len,5),error)
    return(errors)
}

error_eta1=multiclass_perceptron(train.data, train.label,test.data,test.label,0.01)

error_eta2=multiclass_perceptron(train.data, train.label,test.data,test.label,0.09)

## Plot 
plot(x=error_eta1[,1],y=error_eta1[,2], type="l",ylim=c(0,100), col="blue", xlab="number of data", ylab="error")
lines(x=error_eta2[,1],y=error_eta2[,2], type="l", lty=2, lwd=2, col="red")
legend("topright",inset=.05, lty=c(1, 2),  c("Eta 0.01","Eta 0.09"), col=c("blue", "red"))
