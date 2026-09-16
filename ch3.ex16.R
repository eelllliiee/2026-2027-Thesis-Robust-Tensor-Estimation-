
# ----- Exercise 3.16 ----- 

# Write two TTM functions ttm_permute and ttm for any mode of a d-way tensor. The ttm_permute function should perform an explicit permutation of the input tensor in order to perform the TTM with a single matrix multiplication (followed by an explicit permutation of the result). The ttm function should avoid explicit permutations and perform a sequence of matrix multiplications based on the internal structure of the input and output tensors, as in Algorithm 3.1.

# NAIVE APPROACH page 60 
# input: X, a tensor. U, a matrix. k, the mode 
# output: Y, the TTM tensor. 
ttm_permute <- function(X_tens, U_mat, k_mode) {
  
  # dimensions of X 
  dsx <- dim(X_tens)
  ndsx <- length(dim(X_tens))
  # dimensions of U 
  dsu <- dim(U_mat)
  
  # 1 permute mode k to front 
  p_X <- aperm(X_tens, c(k_mode, (1:ndsx)[-k_mode]))
  # 2 mode k unfolding of permuted tensor 
  X_k <- array(p_X, c(dsx[k_mode], prod(dsx[-k_mode])))
  # 3 matrix matrix multiply 
  Y_prod <- U_mat %*% X_k
  # 4 reshape resulting matrix into a tensor 
  Y_block <- array(Y_prod, c(dsu[1], dsx[-k_mode]))
  # 5 permute tensor back 
  inds <- 1:ndsx
  print(c(append(inds[-1], inds[1], after = k_mode)))
  Y_tens <- aperm(Y_block, c(append(inds[-1], inds[1], after = k_mode)))
  
  return(Y_tens)
}

# input: X, a tensor. U, a matrix. k, the mode 
# output: Y, the TTM tensor. 
ttm <- function(X_tens, U_mat, k_mode) {
  
  # dimensions of X 
  dsx <- dim(X_tens)
  ndsx <- length(dim(X_tens))
  # dimensions of U
  dsu <- dim(U_mat)
  
  # mode 1 
  if (k_mode == 1) { 
    # get mode 1 unfolding of X 
    X_mod1 <- array(X_tens, c(dsx[1], prod(dsx[-1])))
    # get product 
    matrix_prod <- U_mat %*% X_mod1
    # reshape product
    Y <- array(matrix_prod, c(dsu[1], dsx[-1]))
  }
  
  # any other mode 
  else {
   Mk <- prod(dsx[1:(k_mode - 1)]) # before mode k 
   Pk <- ifelse(k_mode < ndsx, dsx[(k_mode + 1):ndsx], 1) # after mode k 
   Xbar <- array(X_tens, c(Mk, dsx[k_mode], Pk)) # reshape 
   # batched matrix-matrix multiplication
   Ybar <- list() # initialize, idk the dimensions yet 
   for (i in 1:Pk) {
     Ybar[[i]] <- Xbar[ , , i] %*% t(U_mat) # build slices 
   }
   Ybar <- simplify2array(Ybar) # make list into array 
   # ifelse to avoid NA when k is the last mode and k + 1 does not exist
   reshape_inds_beginning <- dsx[1:(k_mode - 1)]
   reshape_inds_middle <- dsu[1]
   reshape_inds_end <- ifelse(k_mode == ndsx, 
                              c(dsx[1:(k_mode - 1)], dsu[1]),
                              c(dsx[1:(k_mode - 1)], dsu[1], dsx[(k_mode + 1):ndsx]))
   Y <- array(Ybar, c(reshape_inds_beginning,
                      reshape_inds_middle, 
                      reshape_inds_end ))
  }
  
  return(Y)
}

# ----- testing ------

X_test <- array(c(1:8), rep(2, 3))
X_test
U_test <- array(c(1:4), rep(2, 2))
U_test

# naive 
ttm_permute(X_test, U_test, 1)
ttm_permute(X_test, U_test, 2)
ttm_permute(X_test, U_test, 3)

# algorithm 3.1 
ttm(X_test, U_test, 1)
ttm(X_test, U_test, 2)
ttm(X_test, U_test, 3)

# with rTensor package 
Xt <- as.tensor(X_test)
rTensor::ttm(Xt, U_test, 1)
rTensor::ttm(Xt, U_test, 2)
rTensor::ttm(Xt, U_test, 3)






