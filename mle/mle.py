import numpy as np

# logistic regression loss
# -loglikelihood + penalty
def log(output, y, beta):
    """
        output: output of model
        y: label (n,)
    
    """
output = model(input)
loss = -np.sum(np.log(output[y==1])) + gamma//2 * (beta.norm**2)

np.linag.norm(beta)**2



# nll = -np.log(y_pred[y == 1]).sum() - np.log(1 - y_pred[y == 0]).sum()

# gradient

def gradient(y, y_pred):

    # sigma'(beta) = sigma(y_pred)*(1-sigma(y_pred))*x
    pass


