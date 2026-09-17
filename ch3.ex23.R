
library(rTensor)

# algorithm 3.2 

# input: X_tens, a tensor. A_list, a list of matrices of length d. k_mode, the mode 
# output: the MTTKRP 

mttkrp <- function(X_tens, A_list, k_mode) {
  
  # dimensions of X 
  dsx <- dim(X_tens)
  ndsx <- length(dsx)
  # dimensions of A 
  dsa <- length(A_list)
  
  # Case 1: KRP of "lower" modes
  if (k_mode > 1) {
    Mk <- prod(dsx[1:(k_mode - 1)]) # note: no edge case where k = 1
    # handle edge case where only first matrix in the list selected
    # note: using KR function from package rTensor 
    if (k_mode == 2) {
      KL <- A_list[[1]] # the first matrix alone 
    } else {
      KL <- rTensor::khatri_rao_list(A_list[(k_mode - 1):1])
    }
  }
  
  # Case 2: KRP of "upper nodes" 
  if (k_mode < ndsx) {
    Pk <- prod(dsx[(k_mode + 1):ndsx])
    # handle edge case where only last matrix in the list selected
    if (k_mode == (ndsx - 1)) {
      KU <- A_list[[ndsx]] # the last matrix 
    } else {
      KU <- rTensor::khatri_rao_list(A_list[ndsx:(k_mode + 1)])
    }
  }
  
  if (k_mode == 1) {   # edge case 1: mode is 1 
    a <- dsx[1] 
    b <- prod(dsx[-1]) 
    Xbar <- array(X_tens, c(a, b)) # mode 1 unfolding 
    B <- Xbar %*% KU # matrix mult 
  } else if (k_mode == ndsx) { # OR edge case 2: mode is last  
    a <- prod(dsx[-ndsx])
    b <- dsx[ndsx]
    Xbar <- array(X_tens, c(a, b)) # transpose of mode-d unfolding 
    B <- t(Xbar) %*% KL # matrix mult 
  } else {
    Xbar <- array(X_tens, c(Mk, dsx[k_mode], Pk)) # reshape to 3-way tensor 
    Y <- ttm(Xbar, t(KU), 3) # TTM mode-3
    # initialise list since dimensions are unknown 
    B <- list() # batched matrix-vector multiplication 
    for (i in 1:dim(A_list[[1]])[2]) {
      B[[i]] <- t(Y[ , , i]) %*% KL[ , i]  
    }
    # stack vectors in list as column vectors of matrix 
    B <- do.call(cbind, B)
  }
  
    return(B)
}

# equivalent in rTensor package: ttl 

# test objects 
X_test <- array(c(1:8), rep(2, 3))
X_test
A_test <- list("A1" = matrix(1:4, nrow = 2), 
               "A2" = matrix(5:8, nrow = 2), 
               "A3" = matrix(9:12, nrow = 2))
A_test

mttkrp(X_test, A_test, 1)
mttkrp(X_test, A_test, 2)
mttkrp(X_test, A_test, 3)

