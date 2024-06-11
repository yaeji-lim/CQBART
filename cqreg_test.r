library(Rcpp)
library(RcppEigen)
library(MASS)
library(QuantPsyc)
library(RcppArmadillo)

data(Boston)
names(Boston)
y = log(Boston$medv)
y = y- mean(y)
X = Boston[,-14]
X =Make.Z(X)


sourceCpp("cqreg.cpp")

fit=cqr_lasso(X, y, K=9)
