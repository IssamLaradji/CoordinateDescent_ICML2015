ICML2015_GaussSouthwell CoordinateDescent
-----------------------------------------

The corresponding paper (http://arxiv.org/abs/1506.00552) presents new analysis of the coordinate descent algorithm that uses the Gauss Southwell (GS) rule, which can be much faster than random selection; it shows that using exact coordinate optimization and/or lipschitz constants can improve the GS convergence rate. The paper also contains analysis on using fast approximation techniques for the GS rule and analysis for different proximal-gradient GS rules.
  
  
Using the code
--------------


The code serves two main purposes, 

- to generate the experimental results for the coordinate descent algorithms presented in the paper;
including, but not limited to, random selection, non-uniform lipchitz sampling,  and various gauss-southwell rules; and


- to be accessible for users who would like to apply these algorithms on other datasets - the code can be used in a similar way as the algorithms hosted in scikit-learn repository.
