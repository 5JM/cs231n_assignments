# -*- coding: utf-8 -*-
import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1] # 10
  num_train = X.shape[0] # 500
  # dW 는 loss function을 W에 대해서 미분한것.
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    #print(scores.shape) # (10, )
    correct_class_score = scores[y[i]]
    #loss function미분을 해주는데, max부분을 margin > 0
    #이런식으로 나눠서 코딩
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # 정답일땐, 빼주고
        # 아닐땐 class score를 더해준다.
        dW[:,y[i]]-= X[i]
        dW[:,j]+=X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  
  # 1/N을 해주는 부분
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  # L2 regularization
  loss += reg * np.sum(W * W) 
  #loss를 미분해주는거니깐, 앞에 2곱해주고 W만 나온다.
  dW = dW + reg * 2 * W
  
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # compute the loss and the gradient
  num_classes = W.shape[1] # 10
  num_train = X.shape[0] # 500

  scores = X.dot(W) # (num_train, num_classes) 500 * 10
  correct_class_score = scores[range(num_train),y] # (num_train, )

  margin = np.maximum(0, scores - correct_class_score.reshape(num_train,1) + 1)
  margin[range(num_train), y] = 0 # 이미 정답인 경우는 고려하지 않는다, 따라서 0으로 초기화

  loss = np.sum(margin) / num_train
  loss += reg * np.sum(W * W) # l2정규화 
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margin[margin>0] = 1
  valid_margin_count = np.sum(margin, axis=1)
  # Subtract in correct class (-s_y)
  margin[range(num_train), y] -= valid_margin_count
  # 1/N
  dW = (X.T).dot(margin) / num_train
  # loss를 미분
  dW = dW + reg * 2 * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
