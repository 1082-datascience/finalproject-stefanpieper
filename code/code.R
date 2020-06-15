suppressWarnings(library(corrplot))
suppressWarnings(library(randomForest))
suppressWarnings(library(rpart))
suppressWarnings(library(glmnet))


except <- c("raw.train", "raw.test", "raw.all")
rm(list=setdiff(ls(), except))

#### 0. Code Line ####

# Rscript code/code.R --runtype demo --fold 5 --train data/train.csv --test data/test.csv --report results/report.csv --predict results/predict.csv

# read parameters
args = commandArgs(trailingOnly=TRUE)
if (length(args) < 6) {
  stop("USAGE: Rscript hw5_yourID.R --runtype demo/short/long --fold positive integer --train file 
       --test file --report file --predict file", call.=FALSE)
}


# parse parameters
i <- 1 
while(i < length(args)){
  if (args[i] == "--runtype"){
    runtype <- args[i+1]
    if (!runtype %in% c("demo", "short", "long")){
      stop("Please provide correct runtype for rotationForest ('demo' for using old output, 'short' for ntree=c(5, 10), 
           'long' for ntree=c(5, 10, 15, 20, 25, 30, 50, 70, 100)")
    }
    i <- i + 1
  }else if(args[i] == "--fold"){
    nfolds <- as.numeric(args[i+1])
    i <- i + 1
  }else if(args[i] == "--train"){
    train_file <- args[i + 1]
    i <- i + 1
  }else if(args[i] == "--test"){
    test_file <- args[i + 1]
    i <- i + 1
  }else if(args[i] == "--report"){
    report_file <- args[i + 1]
    i <- i + 1
  }else if(args[i] == "--predict"){
    predict_file <- args[i + 1]
    i <- i + 1
  }else{
    stop(paste("Unknown flag", args[i]), call.=FALSE)
  }
  i <- i + 1
}

# read data
raw.train <- read.csv(train_file, stringsAsFactors = FALSE)
raw.test <- read.csv(test_file, stringsAsFactors = FALSE)

if (substr(report_file, nchar(report_file)-3, nchar(report_file)) == ".csv") report_file <- substr(report_file, 1, nchar(report_file)-4)
if (substr(predict_file, nchar(predict_file)-3, nchar(predict_file)) == ".csv") predict_file <- substr(predict_file, 1, nchar(predict_file)-4)

#### 1. Data ####

## code for manual usage
if (!exists("runtype")){
  nfolds <- 5
  raw.train <- read.csv("data/train.csv", header=T)
  raw.test <- read.csv("data/test.csv", header=T)
}

raw.test <- cbind(raw.test[,1], NA, raw.test[,-1])
colnames(raw.test) <- colnames(raw.train)
pred.rotation <- pred.random <- pred.lasso <- pred.zeros <- raw.test[,c(1,2)]


#### 2. Exploration ####

# only runs manually
if (!exists("runtype")){
  
  corrplot2 <- function(Dt1, Dt2=NULL, test=F, diag=F, coef.col="black",
                        title="", txtsize=1, corsize=1, txtpos="td", sig.level=0.05, ...){
    col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
    par(family="serif") ##MIGHT CAUSE TROUBLE
    if (is.null(Dt2)) Dt <- as.matrix(cor(Dt1))
    if (!is.null(Dt2)) Dt <- as.matrix(cor(Dt1,Dt2))
    
    if (is.null(coef.col)){
      if (test == T){
        corrplot(Dt, method="color", col=col(200),
                 type="upper", tl.col="black", tl.srt=45,
                 p.mat = cor.mtest(as.matrix(Dt))$p, sig.level = sig.level, insig = "blank",
                 diag=diag, title = title,
                 tl.cex = txtsize, number.cex = corsize, tl.pos = txtpos)}
      else {
        corrplot(Dt, method="color", col=col(200),
                 type="upper", tl.col="black", tl.srt=45, diag=diag,
                 title = title, tl.cex = txtsize, number.cex = corsize, tl.pos = txtpos)
      }
    }
    else {
      if (test == T){
        corrplot(Dt, method="color", col=col(200),
                 type="upper", addCoef.col=coef.col, tl.col="black", tl.srt=45,
                 p.mat = cor.mtest(as.matrix(Dt))$p, sig.level = sig.level, insig = "blank",
                 diag=diag, title = title, tl.cex = txtsize, number.cex = corsize, tl.pos = txtpos)}
      else {
        corrplot(Dt, method="color", col=col(200),
                 type="upper", tl.col="black", addCoef.col=coef.col, tl.srt=45, diag=diag,
                 title = title, tl.cex = txtsize, number.cex = corsize, tl.pos = txtpos)
      }
    }
  }
  
  
  corrplot2(raw.train[,-c(1,2)], test=F, addCoef.col=F, coef.col=NULL)
  corrplot2(raw.train[,3:10], test=F, addCoef.col=F, coef.col=NULL)
  
}

#### 3. Rotation Forest ####

RotationForest <- function(x, y, K=round(ncol(x)/3, 0), ntree=10, verbose=FALSE, ...){
  
  if (!(is.numeric(y)))
    y <- unname(unlist(y))
  if (K >= ncol(x))
    stop("K should not be greater than or equal to the number of columns in x")
  if (all(sapply(x, is.numeric)) == FALSE)
    stop("All features in x need to be numeric")
  if (K > floor(ncol(x)/2))
    stop(cat("The maximum K for your data is", floor(ncol(x)/2)))
  if (K == 0)
    stop("The minimum K is 1")
  
  # subsets of
  subsetsizes <- rep(floor(ncol(x)/K), K)
  
  # randomly distribute leftover columns over subsets
  leftover <- rep(1, ncol(x) - sum(subsetsizes))
  leftover <- c(leftover, rep(0, K - length(leftover)))
  leftover <- leftover[sample(1:length(leftover), length(leftover), FALSE)]
  subsetsizes <- subsetsizes + leftover
  
  if (verbose == TRUE){
    cat("Number of subsets: ", K, ". Number of variables per respective subset: ", sep="")
    cat(subsetsizes, "\n", sep=" ")
  }
  
  fact <- factor(rep(1:K, subsetsizes))
  Rotation <- list()
  fit <- list()
  subsets <- list()
  SelectedClass <- list()
  IndependentClassSubset <- list()
  IndependentClassSubsetBoot <- list()
  pca <- list()
  loadings <- list()
  predicted <- matrix(NA, nrow=nrow(x), ncol=ntree)
  
  for (i in 1:ntree){
    subsets[[i]] <- list()
    SelectedClass[[i]] <- list()
    IndependentClassSubset[[i]] <- list()
    IndependentClassSubsetBoot[[i]] <- list()
    pca[[i]] <- list()
    loadings[[i]] <- list()
    
    # features for each subset
    features <- split(sample(1:ncol(x), ncol(x), FALSE), fact)
  
    for (j in 1:K){ 
      
      # create subsets with randomly selected features
      subsets[[i]][[j]] <- data.frame(x[, features[[j]]], y)
      
      # select class for response variable
      SelectedClass[[i]][[j]] <- as.integer(sample(levels(as.factor(y)), 1))
      
      # take values of subset[[i]][[j]] where response variable has selected class; take bootstrap sample 75% of data
      IndependentClassSubset[[i]][[j]] <- subsets[[i]][[j]][subsets[[i]][[j]]$y == SelectedClass[[i]][[j]], ]
      IndependentClassSubsetBoot[[i]][[j]] <- IndependentClassSubset[[i]][[j]][sample(1:dim(IndependentClassSubset[[i]][[j]])[1], round(0.75 * nrow(IndependentClassSubset[[i]][[j]])), replace = TRUE), ]
    
      # perform PCA
      pca[[i]][[j]] <- prcomp(subset(IndependentClassSubsetBoot[[i]][[j]], select=-y))
      
      # extract loadings from PCA
      loadings[[i]][[j]] <- pca[[i]][[j]]$rotation
      colnames(loadings[[i]][[j]]) <- dimnames(loadings[[i]][[j]])[[1]]
      loadings[[i]][[j]] <- data.frame(dimnames(loadings[[i]][[j]])[[1]], loadings[[i]][[j]])
      colnames(loadings[[i]][[j]])[1] <- "rowID"
    }
    
    # build rotation matrix
    Rotation[[i]] <- Reduce(function(x, y) merge(x, y, by = "rowID", all = TRUE), loadings[[i]])
    Rotation[[i]][is.na(Rotation[[i]])] <- 0
    Rotation[[i]] <- Rotation[[i]][order(match(Rotation[[i]]$rowID, colnames(x))), order(match(colnames(Rotation[[i]]), colnames(x)))]
    rownames(Rotation[[i]]) <- Rotation[[i]]$rowID
    Rotation[[i]]$rowID <- NULL
    
    # project x (features) on rotation matrix
    final <- data.frame(as.matrix(x) %*% as.matrix(Rotation[[i]]), y)
    fit[[i]] <- rpart(y ~ ., method="class", data=final, ...)
  }
  output <- list(models=fit, loadings=Rotation, columnnames=colnames(x))
  class(output) <- "rotationForest"
  return(output)
}

predict.RotationForest <- function(models, newdata, type=c("prob", "class"), all=FALSE, round.results=TRUE){
  if (class(models) != "rotationForest")
    stop("Model not of class rotationForest")
  if (!type[1] %in% c("prob", "class"))
    stop("Provide either 'prob' or 'class' for type")
  
  type <- type[1]
  len <- length(models$models)
  preds <- matrix(NA, ncol=len, nrow=nrow(newdata))
  final_pred <- NULL
  
  # predict the response for every tree
  for (i in 1:length(models$models)){
    rotated_data <- data.frame(as.matrix(newdata) %*% as.matrix(models$loadings[[i]]))
    pred <- predict(models$models[[i]], rotated_data, type=type)
    for (j in 1:nrow(pred)){
      if (type == "class") preds[j,i] <- pred
      if (type == "prob") preds[j,i] <- pred[j,2]
    }
  }
  
  if (all == TRUE){
    return(preds)
  } else{
    # return mean probability over all trees
    if (type == "prob") {
      output <- apply(preds, 1, function(x) mean(x))
      if (round.results == TRUE) output <- round(output)
    }  
    # return most frequently predicterd class over all trees
    if (type == "class") {
      output <- as.integer(unlist(apply(preds, 1, function(x) names(which.max(table(x))))))
    }
  }
  
  return(output)
}

cv.RotationForest <- function(x, y, x_test=NULL, y_test=NULL, nfolds=5, type.measure=c("accuracy"), round.results=FALSE, test=FALSE,
                              K=round(ncol(x)/3, 0), ntree=10, verbose=FALSE, ...){
  
  folds <- sample(rep(c(1:nfolds), each=ceiling(nrow(x)/nfolds)))
  folds <- folds[1:nrow(x)]
  if ((!is.null(x_test) && !is.null(y_test)) || test == TRUE) {
    perf <- setNames(data.frame(matrix(c(paste0("fold", rep(1:nfolds)), "ave.", rep(NA, 3*(nfolds+1))), ncol=4, nrow=nfolds+1), stringsAsFactors=F), c("set", "training", "validation", "test"))
  } else { 
    perf <- setNames(data.frame(matrix(c(paste0("fold", rep(1:nfolds)), "ave.", rep(NA, 2*(nfolds+1))), ncol=3, nrow=nfolds+1), stringsAsFactors=F), c("set", "training", "validation"))
  }
  model <- list()
  
  for (i in 1:nfolds){
    
    if (verbose == TRUE) cat("Fold ", i, "\n")
    
    if (test == TRUE){
      j <- ifelse(i == nfolds, 1, i+1)
      x_train <- x[!folds %in% c(i, j),]
      y_train <- y[!folds %in% c(i, j),]
      x_valid <- x[folds == i,]
      y_valid <- y[folds == i]
      x_test <- x[folds == j,]
      y_test <- y[folds == j]
    } else {
      x_train <- x[folds != i,]
      y_train <- y[folds != i]
      x_valid <- x[folds == i,]
      y_valid <- y[folds == i]
    }
    
    model[[i]] <- RotationForest(x_train, y_train, K, ntree ,verbose)
    
    pred.train <- predict.RotationForest(model[[i]], x_train, type="prob")
    pred.valid <- predict.RotationForest(model[[i]], x_valid, type="prob")
    
    perf[i, "training"] <- sum(pred.train == y_train) / nrow(x_train)
    perf[i, "validation"] <- sum(pred.valid == y_valid) / nrow(x_valid)
    
    # predict on test data if available
    if (!is.null(x_test)){
      pred.test <- predict.RotationForest(model[[i]], x_test, type="prob")
      if (!is.null(y_test)){
        perf[i, "test"] <- sum(pred.test == y_test) / nrow(y_test)
      }
    }
  }
  
  # averages of train and validation errors
  for (k in 2:4){
    perf[nfolds+1, k] <- mean(as.numeric(perf[1:nfolds, k]))
  }
  
  # round results
  if (round.results == TRUE){
    for (m in 2:4){
      for (n in 1:(nfolds+1)){
        perf[n,m] <- round(as.numeric(perf[n,m]), 2)
      }
    }
  }
  
  # return results
  if (exists("pred.test")) {
    return(list(pred.test=pred.test, models=model, performance=perf))
  } else {
    return(list(models=model, performance=perf))
  }
}

vary.RotationForest <- function(x, y, x_test=NULL, y_test=NULL, plot.results=TRUE, nfolds=5, type.measure=c("accuracy"), round.results=FALSE, test=FALSE,
                                K=c(round(ncol(x)/3, 0), round(ncol(x)/6, 0), round(ncol(x)/9, 0)), ntree=c(50, 100), verbose=FALSE, save.file=NULL, ...){
  output <- list()
  for (k in K){
    for (l in ntree){
      
      if (verbose == TRUE) cat("\nK: ", k, "\t ntree: ", l, "\n")
      
      output[[length(output)+1]] <- list()
      output[[length(output)]]$K <- k
      output[[length(output)]]$ntree <- l
      #output[[length(output)]]$result <- cv.RotationForest(raw.train[,-c(1,2)], raw.train[,2], K=k, ntree=l, nfolds=5, round=T, verbose=T)
      output[[length(output)]]$result <- cv.RotationForest(x, y, x_test, y_test, nfolds, type.measure, round.results, test, K=k, ntree=l, verbose, ...)
      
      output[[length(output)]]$mse.train <- output[[length(output)]]$result$performance[nfolds+1, "training"]
      output[[length(output)]]$mse.valid <- output[[length(output)]]$result$performance[nfolds+1, "validation"]
      if (!is.null(y_test) || test == TRUE) output[[length(output)]]$mse.test <- output[[length(output)]]$result$performance[nfolds+1, "test"]
      
      if (!is.null(save.file)){
        if (verbose == TRUE) cat("Saving... \n")
        tmp.env <- new.env(); assign("output", output, pos=tmp.env)
        save(list = ls(all.names=TRUE, pos=tmp.env), envir=tmp.env, file=paste0(save.file,".RData")); rm(tmp.env)
      }
    }
  }
  
  if (is.null(y_test) && test == FALSE) {
    data <- setNames(data.frame(matrix(rep(NA, length(output)*4), ncol=4)), c("K", "ntree", "mse.train", "mse.validation"))
  } else {
    data <- setNames(data.frame(matrix(rep(NA, length(output)*5), ncol=5)), c("K", "ntree", "mse.train", "mse.validation", "mse.test"))
  }
  
  for (i in 1:length(output)){
    data[i, 1] <- as.numeric(output[[i]]$K)
    data[i, 2] <- as.numeric(output[[i]]$ntree)
    data[i, 3] <- as.numeric(output[[i]]$mse.train)
    data[i, 4] <- as.numeric(output[[i]]$mse.valid)
    if (ncol(data) == 5) data[i, 5] <- as.numeric(output[[i]]$mse.test)
  }

  # predict best model on test set
  if (!is.null(x_test)){
    best <- 0
    for (i in 1:nrow(data)){
      if (best == 0 || data[i, 4] > data[best, 4]) best <- i
    }
    best.K <- data[best, 1]
    best.ntree <- data[best, 2]
    if (verbose == TRUE) cat("\nBest K: ", best.K, "\t Best ntree: ", best.ntree, "\n\n")
    best.model <- RotationForest(raw.train[,-c(1,2)], raw.train[,2], K=best.K, ntree=best.ntree)
    pred <- predict.RotationForest(best.model, raw.test[,-c(1,2)])
  }
  
  output <-list(prediction=pred, nfolds=nfolds, best.K=best.K, best.ntree=best.ntree, best.model=best.model, summary=data, models=output)
  
  if (!is.null(save.file)){
    if (verbose == TRUE) cat("Saving Complete Result... \n")
    tmp.env <- new.env(); assign("output", output, pos=tmp.env)
    save(list = ls(all.names=TRUE, pos=tmp.env), envir=tmp.env, file=paste0(save.file,".RData")); rm(tmp.env)
  }
  
  if (plot.results == TRUE){
    plot.RotationForest(output)
  }
  
  return(output)
}

plot.RotationForest <- function(model, pos="topright", col=NULL){
  
  data <- model$summary
  nfolds <- model$nfolds
  L <- unique(data$ntree)
  if (is.null(col)) col <- rainbow(length(L))
  
  for (i in 3:ncol(data)){
    plot(data[data$ntree == L[1], 1], data[data$ntree == L[1],i], col=col[1], lwd=2, type="b", main=paste(nfolds, " folds ", c("training","validation","test")[i-2]), xlab="K", ylab="accuracy", xlim=c(min(data$K), max(data$K)), ylim=as.numeric(c(min(data[,i]), max(data[,i]))))
    for (l in 2:length(L)){
      lines(data[data$ntree == L[l], 1], data[data$ntree == L[l],i], col=col[l], type="b", lwd=2)
    }
    legend(pos, legend=L, col=rainbow(length(L)), cex=0.8, lty=1, title="ntree", lwd=2)
  }
}



if (exists("runtype") && runtype == "long"){
res.rotation <- vary.RotationForest(raw.train[,-c(1,2)], raw.train[,2], x_test=raw.test[,-c(1,2)], K=c(10, 20, 33, 50, 70, 100), 
                           ntree=c(5, 10, 15, 20, 25, 30, 50, 70, 100), nfolds=nfolds, round.results=F, verbose=T)
}

if (exists("runtype") && runtype == "short"){
set.seed(2)
res.rotation <- vary.RotationForest(raw.train[,-c(1,2)], raw.train[,2], x_test=raw.test[,-c(1,2)], 
                                    K=c(10, 20), ntree=c(5, 10), nfolds=nfolds, round.results=F, verbose=T)
}

if (exists("runtype") && runtype == "demo"){
  load("data/rotation_output.RData")
  res.rotation <- get("output"); rm(output)
  plot.RotationForest(res.rotation, pos="bottomleft")
}

best.model <- RotationForest(raw.train[,-c(1,2)], raw.train[,2], K=res.rotation$best.K, ntree=res.rotation$best.ntree, verbose=TRUE)
pred.rotation[,2] <- predict.RotationForest(best.model, raw.test[,-c(1,2)])
write.table(pred.rotation, row.names=F, file=paste0(predict_file, "_rotationForest.csv"), quote=F, sep=",")

#tmp.env <- new.env(); assign(paste0("result_rotation"), res.rotation, pos=tmp.env)
#save(list = ls(all.names=TRUE, pos=tmp.env), envir=tmp.env, file="results/result_rotationForest.RData"); rm(tmp.env)
#pred.rotation[,2] <- res.rotation$prediction



#### 4. Random Forest ####

vary.randomForest <- function(x, y, x_test=NULL, nfolds=5, ntree=10, replace=TRUE, verbose=TRUE){
  folds <- sample(rep(c(1:nfolds), each=ceiling(nrow(raw.train)/nfolds)))
  folds <- folds[1:nrow(raw.train)]

  best <- 0
  perf <- model <- list()
  for (r in replace){
    for (s in 1:length(ntree)){
      
      i <- length(ntree)*r + s
      if (verbose == TRUE) cat("ntree:", ntree[s], "\t replace: ", r, "\n")
      perf[[i]] <- setNames(data.frame(matrix(c(paste0("fold", rep(1:nfolds)), "ave.", rep(NA, 2*(nfolds+1))), ncol=3, nrow=nfolds+1), stringsAsFactors=F), c("set", "training", "validation"))
      
      for (j in 1:nfolds){
        x_train <- x[folds != j,]
        y_train <- y[folds != j]
        x_valid <- x[folds == j,]
        y_valid <- y[folds == j]
        
        model[[i]]<- randomForest(x=x_train, y=y_train, ntree=ntree[s], replace=r)
        
        pred.train <- as.numeric(as.character(predict(model[[i]], x_train, type="response")))
        pred.valid <- as.numeric(as.character(predict(model[[i]], x_valid, type="response")))
        
        perf[[i]][j, "training"] <- sum(pred.train == as.numeric(as.character(y_train))) / nrow(x_train)
        perf[[i]][j, "validation"] <- sum(pred.valid == as.numeric(as.character(y_valid))) / nrow(x_valid)
      }
      
      # averages of train and validation errors
      for (l in 2:4){
        perf[[i]][nfolds+1, l] <- mean(as.numeric(perf[[i]][1:nfolds, l]))
      }
      
      # check which model is the best
      if (best == 0 || perf[[i]][nfolds+1, "validation"] > perf[[best]][nfolds+1, "validation"]){
        best_ntree <- ntree[s]
        best_replace <- r
        best <- i
      }
      
      # round results
      for (m in 2:4){
        for (n in 1:(nfolds+1)){
          perf[[i]][n,m] <- (as.numeric(perf[[i]][n,m]))#, 2)
        }
      }
    }  
  }
  
  # return best model
  best.model <- randomForest(x=x, y=y, ntree=best_ntree, replace=best_replace)
  pred <- as.numeric(as.character(predict(best.model, x_test, type="response")))
  
  return(list(prediction=pred, best.model=best.model, best_ntree=best_ntree, best_replace=best_replace, performance=perf, models=model))
}

res.random <- vary.randomForest(x=raw.train[,-c(1,2)], y=as.factor(raw.train[,2]), x_test=raw.test[,-c(1,2)], nfolds=nfolds, ntree <- c(10, 25, 50, 100, 200), replace <- c(F,T))
pred.random[,2] <- res.random$prediction
write.table(pred.random, row.names=F, file=paste0(predict_file, "_LASSO.csv"), quote=F, sep=",")       

#tmp.env <- new.env(); assign(paste0("result_random"), res.random, pos=tmp.env)
#save(list = ls(all.names=TRUE, pos=tmp.env), envir=tmp.env, file="results/result_randomForest.RData"); rm(tmp.env)



#### 5. LASSO ####

res.lasso <- cv.glmnet(as.matrix(raw.train[,-c(1,2)]), raw.train[,2], family = "binomial", alpha=1, lambda=NULL, nfolds=nfolds)
res.lasso <- glmnet(as.matrix(raw.train[,-c(1,2)]), raw.train[,2], family = "binomial", alpha=1, lambda = res.lasso$lambda.min)

pred.lasso[,2] <- predict(res.lasso, newx=as.matrix(raw.test[,-c(1,2)]), type="response")
write.table(pred.lasso, row.names=F, file=paste0(predict_file, "_randomForest.csv"), quote=F, sep=",")



#### 5. Zero's ####

pred.zeros[,2] <- 0
write.table(pred.lasso, row.names=F, file=paste0(predict_file, "_zeros.csv"), quote=F, sep=",")

