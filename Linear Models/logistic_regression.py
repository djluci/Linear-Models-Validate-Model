'''TODO: Header'''
##Keep these imports as-is
import jax.numpy as jnp
from jax import grad
from jax.nn import sigmoid
from jax import random
from sklearn.preprocessing import StandardScaler
import itertools

class LogisticRegression:
    ''' Logistic regression with option to add interaction terms with their own sparsity penalty.'''

    ##NOTE: penalty and alpha are used starting at step 2, interactions starting at step 3 and interaction_alpha starting at step 4
    def __init__( self , feature_names , step_size=1e-2 , n_iter=5000 , penalty=None , alpha=0. , interactions=False , interaction_alpha=None, scale=True):

        ##NOTE: Each of these comments corresponds to a class field that you are suggested to have. If there's no step number, you'll need them from the first step!

        ##STEP 2 -- Type of penalty.  Options are None, 'l1', 'l2'

        ##STEP 2 -- Penalty weight

        ##STEP 3 -- Flag for including interaction terms

        ##STEP 3 -- Standard scaler - to use for scaling interaction terms

        ##Feature names -- (STEP 3: these should include interaction terms where applicable)

        ##STEP 4 -- Indices of non-interaction features

        ##STEP 4 -- Indices of interaction features (empty if none)

        ##STEP 4 -- Interaction terms penalty weight. If none, set to alpha.

        ##Number of features (STEP 3: This should include interaction terms where applicable)

        ##Number of iterations of gradient descent to use in training

        ##Step size to use when updating parameters in gradient descent

        ##Randomly initialized weights of size self.n_features with mean 0 and variance 1/100.

        ##Randomly initialized bias of size 1, with mean 0 and variance 1/100.

        self.feature_names = feature_names
        self.interactions = interactions
        self.n_features = len(feature_names)
        self.step_size = step_size
        self.n_iter = n_iter
        self.penalty = penalty
        self.alpha = alpha
        self.scaler = StandardScaler() if scale else None
        self.interaction_alpha = interaction_alpha if interaction_alpha is not None else alpha  #  utilizing interaction_alpha if provided

        # Delayed weight initialization, to be set in fit() based on processed feature size
        self.weights = None
        self.bias = None
        self.interaction_indices = [] # storing interaction term indices
        self.original_indices = [] # storing original feature indices

    def process_features(self, X):
        '''STEP 3
        Add interaction terms and apply standard scaling to the processed feature vector (to scale the interaction terms)

        Parameters:
        ----------
        X: ndarray. Shape = [Num samples N, Num features D]
            Collection of input vectors.

        Returns:
        ----------
        A processed collection of input vectors with interaction terms added if self.interactions=True, and with standard scaling applied

        NOTES:
        1. Only fit self.Scaler on the training set. You can do this by only fitting it if self.scaler is None (that way you only fit it once).
        '''
        # Original features
        processed_x = X
        # If interactions are enabled, generate interaction terms
        if self.interactions:
            # Generate pairwise interaction terms
            interaction_terms = []
            for i, j in itertools.combinations(range(X.shape[1]), 2):
                interaction_terms.append((X[:, i] * X[:, j]).reshape(-1, 1))  # Element-wise multiplication
            # Stack interaction terms with original features
            interaction_terms = jnp.hstack(interaction_terms)
            processed_x = jnp.hstack([X, interaction_terms])
            
            # Storing indices for original and interaction terms
            self.original_indices = list(range(X.shape[1]))
            self.interaction_indices = list(range(X.shape[1], processed_x.shape[1]))

        # Only scale the data if scaling is enabled
        if self.scaler:
            if not hasattr(self.scaler, 'mean_'):
                self.scaler.fit(processed_x)
            processed_x = self.scaler.transform(processed_x)

        return processed_x

    def fit(self, X, y):
        '''Fits weights and bias based on features X and labels y.

        Parameters:
        ----------
        X: ndarray. Shape = [Num samples N, Num features D]
            Collection of input vectors.
        y: ndarray. Shape = [Num samples N,]
            True classes corresponding to each input sample (coded as 0 or 1).

        Returns:
        ----------
        Nothing

        NOTES:
        1. Call the update function n_iter times and use it to update the weights and biases
        '''
        # Process features first to determine the final shape
        x_processed = self.process_features(X)

        # Initialize weights based on the processed features shape
        if self.weights is None:
            # Initialize weights to match the number of processed features
            key = random.PRNGKey(0)
            self.weights = random.normal(key, (x_processed.shape[1],)) / 100
            self.bias = random.normal(key, (1,)) / 100

        # Gradient descent to update weights and bias
        params = (self.weights, self.bias)
        for _ in range(self.n_iter):
            params = self.update(params, x_processed, y)

        # Final weights and bias
        self.weights, self.bias = params

    def predict_proba_with_params(self, params, X, processed=False):
        '''Predicts the probability of class=1 corresponding to each row of X (each instance)

        Parameters:
        ----------
        X: ndarray. Shape = [Num samples N, Num features D]
            Collection of input vectors.
        params: tuple.
            Tuple of (weights, biases) where weights is ndarray and has shape = [Num features D,]
            and biases is ndarray and has shape = [1,]

        Returns:
        ----------
        1-D array of predicted probabilities of class=1 for each input feature vector. Shape = [Num samples N,]
        '''
        weights, bias = params
        if not processed:
            X_processed = self.process_features(X)
        else:
            X_processed = X  # If already processed, just use X directly

        logits = jnp.dot(X_processed, weights) + bias
        return sigmoid(logits)

    def predict_proba(self, X):
        '''Predicts the probability of class=1 corresponding to each row of X (each instance)

        Parameters:
        ----------
        X: ndarray. Shape = [Num samples N, Num features D]
            Collection of input vectors.

        Returns:
        ----------
        1-D array of predicted probabilities of class=1 for each input feature vector. Shape = [Num samples N,]
        '''
        proba_class_1 = self.predict_proba_with_params((self.weights, self.bias), X)
        proba_class_0 = 1 - proba_class_1
        return jnp.vstack([proba_class_0, proba_class_1]).T

    def predict(self, X):
        '''Predicts the class corresponding to each row of X (each instance)

        Parameters:
        ----------
        X: ndarray. Shape = [Num samples N, Num features D]
            Collection of input vectors.

        Returns:
        ----------
        1-D array of predicted classes (0 or 1) for each input feature vector. Shape = [Num samples N,]
        '''
        # Probabilities for class 1
        prob_class_1 = self.predict_proba(X)[:, 1]
        # Convert to binary predictions (0 or 1)
        return (prob_class_1 > 0.5).astype(int)

    def binary_cross_entropy_loss(self, y, y_proba):
        '''Computes and returns the binary cross entropy loss

        Parameters:
        ----------
        y: ndarray. Shape = [Num samples N,]
            True classes corresponding to each input sample (coded as 0 or 1).
        y_proba: ndarray. Shape = [Num samples N,]
            Predicted probabilities of class=1 corresponding to each input sample

        Returns:
        ----------
        The binary cross entropy loss
        '''
        y_proba_clipped = jnp.clip(y_proba, 1e-14, 1 - 1e-14)  # Avoid NaNs when taking log
        return -jnp.mean(y * jnp.log(y_proba_clipped) + (1 - y) * jnp.log(1 - y))

    def sparsity_loss(self, w):
        '''STEP 2
        Computes and returns the sparsity loss

        Parameters:
        ----------
        w: ndarray. Shape = [Num features D,]

        Returns:
        ----------
        The sparsity loss
        '''
        #step 2
        if self.penalty == 'l1':
            basic_features_loss = self.alpha * jnp.linalg.norm(w[:len(self.feature_names)], ord=1)  # L1 norm for basic features
        elif self.penalty == 'l2':
            basic_features_loss = self.alpha * jnp.linalg.norm(w[:len(self.feature_names)], ord=2)  # L2 norm for basic features
        else:
            basic_features_loss = 0.
        
        #Step 4
        # Interaction terms
        if self.interactions:
            if self.penalty == 'l1':
                interaction_loss = self.interaction_alpha * jnp.linalg.norm(w[jnp.array(len(self.feature_names)):], ord=1)  # L1 norm for interaction terms
            elif self.penalty == 'l2':
                interaction_loss = self.interaction_alpha * jnp.linalg.norm(w[jnp.array(len(self.feature_names)):], ord=2)  # L2 norm for interaction terms
            else:
                interaction_loss = 0.
        else:
            interaction_loss = 0.
        
        # Combine both losses
        return basic_features_loss + interaction_loss

    def loss(self, params, X, y):
        '''Computes the binary cross entropy loss and sparsity loss (if applicable)

        Parameters:
        ----------
        X: ndarray. Shape = [Num samples N, Num features D]
            Collection of input vectors.
        y: ndarray. Shape = [Num samples N,]
            True classes corresponding to each input sample (coded as 0 or 1).
        params: tuple.
            Tuple of (weights, biases) where weights is ndarray and has shape = [Num features D,]
            and biases is ndarray and has shape = [1,]

        Returns:
        ----------
        The binary cross entropy loss + the sparsity loss (after step 2)
        '''
        weights, bias = params
        y_proba = self.predict_proba_with_params(params, X, processed=True)
        bce_loss = self.binary_cross_entropy_loss(y, y_proba)
        reg_loss = self.sparsity_loss(weights)
        return bce_loss + reg_loss

    def update(self, params, X, y):
        '''Updates and returns the weights and biases

        Parameters:
        ----------
        X: ndarray. Shape = [Num samples N, Num features D]
            Collection of input vectors.
        y: ndarray. Shape = [Num samples N,]
            True classes corresponding to each input sample (coded as 0 or 1).
        params: tuple.
            Tuple of (weights, biases) where weights is ndarray and has shape = [Num features D,]
            and biases is ndarray and has shape = [1,]

        Returns:
        ----------
        The updated weights, the updated bias
        '''
        loss_grad = grad(self.loss)(params, X, y)
        weights, bias = params
        new_weight = weights - self.step_size * loss_grad[0]
        new_bias = bias - self.step_size * loss_grad[1]
        return new_weight, new_bias