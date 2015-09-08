library(tm);                                                                                                              
                                                                                                                          
# Get the corpus for "wheat" and "crude" documents                                                                        
wheat <- Corpus(DirSource("wheat"), readerControl=list(reader=readPlain,language="en_US"));                               
crude <- Corpus(DirSource("crude"), readerControl=list(reader=readPlain,language="en_US"));                               
                                                                                                                          
# Get corresponding train and test sets (70% and 30% respectively)                                                        
lg <- length(wheat);                                                                                                      
wheat.train <- wheat[1:as.integer(lg*0.7)];                                                                               
wheat.test <- wheat[(as.integer(lg*0.7)+1):lg];                                                                           
lg <- length(crude);                                                                                                      
crude.train <- crude[1:as.integer(lg*0.7)];                                                                               
crude.test <- crude[(as.integer(lg*0.7)+1):lg];                                                                           
l1 <- length(wheat.train);                                                                                                
l2 <- length(crude.train);                                                                                                
l3 <- length(wheat.test);                                                                                                 
l4 <- length(crude.test);                                                                                                 
                                                                                                                          
# Merge corpora into one collection                                                                                       
docs <- c(wheat.train, crude.train, wheat.test, crude.test);                                                              
                                                                                                                          
# Pre-processing                                                                                                          
docs.p <- docs;                                                                                                           
docs.p <- tm_map(docs.p, stripWhitespace);                                                                                
docs.p <- tm_map(docs.p, tolower);                                                                                        
docs.p <- tm_map(docs.p, removeWords, stopwords("en"));                                                                   
docs.p <- tm_map(docs.p, removePunctuation);                                                                              
docs.p <- tm_map(docs.p, removeNumbers);                                                                                  
                                                                                                                          
# Create a Document-Term matrix                                                                                           
dtm.mx <- DocumentTermMatrix(docs.p, control=list(weighting=weightTf));                                                   
                                                                                                                          
# Convert the Document-Term matrix into a data frame and append class information                                         
dtm <- as.data.frame(inspect(dtm.mx));                                                                                    
rownames(dtm)<- 1:nrow(dtm.mx);                                                                                           
class <- c(rep("wheat",l1), rep("crude",l2), rep("wheat",l3), rep("crude",l4));                                           
dtm <- cbind(dtm, class);                                                                                                 
last.col <- length(dtm);                                                                                                  
                                                                                                                          
# Prepare data for training and testing the classifier                                                                    
dtm.tr <- dtm[1:(l1+l2), 1:last.col];                                                                                                                                                                                                        
dtm.ts <- dtm[(l1+l2+1):(l1+l2+l3+l4),1:(last.col-1)];                                                                    
                                                                                                                          
rename.terms.in.dtm <- function(dtm) {                                                                                    
    for (i in 1:length(dtm)) {                                                                                            
        cat("replaced to ", paste(colnames(dtm)[i],".t", sep=""), "\n")                                                   
        colnames(dtm)[i] <- paste(colnames(dtm)[i],".t", sep="")                                                          
    } #end for i                                                                                                          
return(dtm)                                                                                                               
}
                                                                                                                          
dtm.tr <- rename.terms.in.dtm(dtm.tr);                                                                                    
dtm.ts <- rename.terms.in.dtm(dtm.ts);                                                                                    
                                                                                                                          
# Identification of informative terms                                                                                     
info.terms <- findFreqTerms(dtm.mx, 10);                                                                                  
cat("Number of features:", length(info.terms) - 1, "\n");                                                                 
                                                                                                                          
rename.terms.in.list <- function(list) {                                                                                  
    for (i in 1:length(list)) {                                                                                           
        #cat("replaced", list[i], "at", i, "with", paste(list[i],".t", sep=""), "\n")                                     
            list[i]<- paste(list[i],".t", sep="")                                                                         
    } #end for i                                                                                                          
#   list <- list[list != "crude.t"];                                                                                      
#   list <- list[list != "wheat.t"];                                                                                      
    return(list)                                                                                                          
}                                                                                                                         
                                                                                                                          
# Rename terms to avoid errors in classification                                                                          
info.terms <- rename.terms.in.list(info.terms);                                                                           
                                                                                                                          
# Train the classifier                                                                                                    
names.tr <- paste(info.terms, collapse='+');                                                                              
class.formula <- as.formula(paste('class.t', names.tr, sep='~'));                                                         
                                                                                                                          
class.ts <- dtm[(l1+l2+1):(l1+l2+l3+l4),last.col];                                                                        
class.tr <- dtm[1:(l1+l2),last.col];                                                                                      
                                                                                                                          
library(rpart)                                                                                                            
dt <- rpart(class.formula, dtm.tr)                                                                                        
#Prediction                                                                                                               
preds.dt <- predict(dt, dtm.ts, type="class")                                                                             
#Construct a confusion matrix:                                                                                            
conf.mx.dt <- table(class.ts, preds.dt)                                                                                   
#Percentage of prediction errors                                                                                          
error.rate.dt <- (sum(conf.mx.dt) - sum(diag(conf.mx.dt))) / sum(conf.mx.dt)                                              
#Evaluation of classifier                                                                                                 
tp.dt <- conf.mx.dt[1,1] #(true positive)                                                                                 
fp.dt <- conf.mx.dt[2,1] #(false positive)                                                                                
tn.dt <- conf.mx.dt[2,2] #(true negative)                                                                                 
fn.dt <- conf.mx.dt[1,2] #(false negative)                                                                                
recall.dt = tp.dt / (tp.dt + fn.dt)                                                                                       
precision.dt = tp.dt / (tp.dt + fp.dt)                                                                                    
f1.dt = 2 * precision.dt * recall.dt / (precision.dt + recall.dt)                                                         
accuracy.dt = (tp.dt + tn.dt) / (tp.dt + fp.dt + tn.dt + fn.dt)                                                           
                                                                                                                          
library(nnet)                                                                                                             
nnet.classifier <- nnet(class.formula, data = dtm.tr, size=2, rang=0.1,decay=5e-4, maxit=200)                             
preds.nn <- predict(nnet.classifier, dtm.ts, type="class")                                                                
conf.mx.nn <- table(class.ts, preds.nn)                                                                                   
error.rate.nn <- (sum(conf.mx.nn) - sum(diag(conf.mx.nn))) / sum(conf.mx.nn)                                              
#Evaluation of classifier                                                                                                 
tp.nn <- conf.mx.nn[1,1] #(true positive)                                                                                 
fp.nn <- conf.mx.nn[2,1] #(false positive)                                                                                                                                                                                                   
tn.nn <- conf.mx.nn[2,2] #(true negative)                                                                                 
fn.nn <- conf.mx.nn[1,2] #(false negative)                                                                                
recall.nn = tp.nn / (tp.nn + fn.nn)                                                                                       
precision.nn = tp.nn / (tp.nn + fp.nn)                                                                                    
f1.nn = 2 * precision.nn * recall.nn / (precision.nn + recall.nn)                                                         
accuracy.nn = (tp.nn + tn.nn) / (tp.nn + fp.nn + tn.nn + fn.nn)                                                           

library(class)                                                                                                            
preds.knn <- knn(dtm.tr[, info.terms], dtm.ts[, info.terms], class.tr, k=1)                                               
conf.mx.knn <- table(class.ts, preds.knn)                                                                                 
error.rate.knn <- (sum(conf.mx.knn) - sum(diag(conf.mx.knn))) / sum(conf.mx.knn)                                          
#Evaluation of classifier                                                                                                 
tp.knn <- conf.mx.knn[1,1] #(true positive)                                                                               
fp.knn <- conf.mx.knn[2,1] #(false positive)                                                                              
tn.knn <- conf.mx.knn[2,2] #(true negative)                                                                               
fn.knn <- conf.mx.knn[1,2] #(false negative)                                                                              
recall.knn = tp.knn / (tp.knn + fn.knn)                                                                                   
precision.knn = tp.knn / (tp.knn + fp.knn)                                                                                
f1.knn = 2 * precision.knn * recall.knn / (precision.knn + recall.knn)                                                    
accuracy.knn = (tp.knn + tn.knn) / (tp.knn + fp.knn + tn.knn + fn.knn)                                                    
                                                                                                                          
library(e1071)                                                                                                            
svm.classifier <- svm(class.formula, dtm.tr)                                                                              
preds.svm <- predict(svm.classifier, dtm.ts)                                                                              
conf.mx.svm <- table(class.ts, preds.svm)                                                                                 
error.rate.svm <- (sum(conf.mx.svm) - sum(diag(conf.mx.svm))) / sum(conf.mx.svm)                                          
#Evaluation of classifier                                                                                                 
tp.svm <- conf.mx.svm[1,1] #(true positive)                                                                               
fp.svm <- conf.mx.svm[2,1] #(false positive)                                                                              
tn.svm <- conf.mx.svm[2,2] #(true negative)                                                                               
fn.svm <- conf.mx.svm[1,2] #(false negative)                                                                              
recall.svm = tp.svm / (tp.svm + fn.svm)                                                                                   
precision.svm = tp.svm / (tp.svm + fp.svm)                                                                                
f1.svm = 2 * precision.svm * recall.svm / (precision.svm + recall.svm)                                                    
accuracy.svm = (tp.svm + tn.svm) / (tp.svm + fp.svm + tn.svm + fn.svm)                                                    
                                                                                                                          
library(RWeka)                                                                                                            
NB<-make_Weka_classifier("weka/classifiers/bayes/NaiveBayes")                                                             
nb.classifier<-NB(class.formula, dtm.tr)                                                                                  
preds.nb<-predict(nb.classifier, dtm.ts)                                                                                  
conf.mx.nb<-table(class.ts, preds.nb)                                                                                     
error.rate.nb <- (sum(conf.mx.nb) - sum(diag(conf.mx.nb))) / sum(conf.mx.nb)                                              
#Evaluation of classifier                                                                                                 
tp.nb <- conf.mx.nb[1,1] #(true positive)                                                                                 
fp.nb <- conf.mx.nb[2,1] #(false positive)                                                                                
tn.nb <- conf.mx.nb[2,2] #(true negative)                                                                                 
fn.nb <- conf.mx.nb[1,2] #(false negative)                                                                                
recall.nb = tp.nb / (tp.nb + fn.nb)                                                                                       
precision.nb = tp.nb / (tp.nb + fp.nb)                                                                                    
f1.nb = 2 * precision.nb * recall.nb / (precision.nb + recall.nb)                                                         
accuracy.nb = (tp.nb + tn.nb) / (tp.nb + fp.nb + tn.nb + fn.nb)                                                           
                                                                                                                          
cat("\nNB: Conf. Matrix:", conf.mx.nb, "\n", "Error rate: ", error.rate.nb, "\n", "Accuracy: ", accuracy.nb, "\n", "Precision: ", precision.nb, "\n", "Recall: ", recall.nb, "\n", "F1: ", f1.nb, "\n");
                                                                                                                          
cat("\nSVM: Conf. Matrix:", conf.mx.svm, "\n", "Error rate: ", error.rate.svm, "\n", "Accuracy: ", accuracy.svm, "\n", "Precision: ", precision.svm, "\n", "Recall: ", recall.svm, "\n", "F1: ", f1.svm, "\n");
                                                                                                                          
cat("\nKNN: Conf. Matrix:", conf.mx.knn, "\n", "Error rate: ", error.rate.knn, "\n", "Accuracy: ", accuracy.knn, "\n", "Precision: ", precision.knn, "\n", "Recall: ", recall.knn, "\n", "F1: ", f1.knn, "\n");
                                                                                                                          
cat("\nNN: Conf. Matrix:", conf.mx.nn, "\n", "Error rate: ", error.rate.nn, "\n", "Accuracy: ", accuracy.nn, "\n", "Precision: ", precision.nn, "\n", "Recall: ", recall.nn, "\n", "F1: ", f1.nn, "\n");                                     
                                                                                                                          
cat("\nDT: Conf. Matrix:", conf.mx.dt, "\n", "Error rate: ", error.rate.dt, "\n", "Accuracy: ", accuracy.dt, "\n", "Precision: ", precision.dt, "\n", "Recall: ", recall.dt, "\n", "F1: ", f1.dt, "\n");
                                                                                                                          
