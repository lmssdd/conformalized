import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_pinball_loss
import pandas as pd

class ConfGradientBoostingRegressor(HistGradientBoostingRegressor):
    """Conformalized Histogram-based Gradient Boosting Regression Tree.

    This estimator is based on
    :class:`HistGradientBoostingRegressor<sklearn.ensemble.HistGradientBoostingRegressor>`
    with additional methods for optimization of the parameters through cross 
    validation and conformalization of the prediction handling heteroscedasticity.

    This estimator has native support for missing values (NaNs). During
    training, the tree grower learns at each split point whether samples
    with missing values should go to the left or right child, based on the
    potential gain. When predicting, samples with missing values are
    assigned to the left or right child consequently. If no missing values
    were encountered for a given feature during training, then samples with
    missing values are mapped to whichever child has the most samples.
    
    The model only performs Conformalized Quantile Regression based on [1], the
    conformalization is performed on the predicted quantiles separately and not 
    on the intervals. 
    An additional cluster series is passed to the conformalization procedures to 
    calculate the conformalization correction separately for each cluster index of 
    the series, the clusters should be separated on the basis of the expecteded 
    amplitude of the prediction intervals, therefore separating data on the basis of 
    expected uncertainty. The clusters are passed explicitly in this formulation 
    as an altrnative to the implicit treatment proposed in [2]

    Parameters
    ----------
    quantiles : array-like of float, default=[0.5]
        This parameter specifies the quantiles to be estimated, each of these
        must be between 0 and 1.
    learning_rate : float, default=0.1
        The learning rate, also known as *shrinkage*. This is used as a
        multiplicative factor for the leaves values. Use ``1`` for no
        shrinkage.
    max_iter : int, default=100
        The maximum number of iterations of the boosting process, i.e. the
        maximum number of trees.
    max_leaf_nodes : int or None, default=31
        The maximum number of leaves for each tree. Must be strictly greater
        than 1. If None, there is no maximum limit.
    max_depth : int or None, default=None
        The maximum depth of each tree. The depth of a tree is the number of
        edges to go from the root to the deepest leaf.
        Depth isn't constrained by default.
    min_samples_leaf : int, default=20
        The minimum number of samples per leaf. For small datasets with less
        than a few hundred samples, it is recommended to lower this value
        since only very shallow trees would be built.
    l2_regularization : float, default=0
        The L2 regularization parameter. Use ``0`` for no regularization
        (default).
    max_bins : int, default=255
        The maximum number of bins to use for non-missing values. Before
        training, each feature of the input array `X` is binned into
        integer-valued bins, which allows for a much faster training stage.
        Features with a small number of unique values may use less than
        ``max_bins`` bins. In addition to the ``max_bins`` bins, one more bin
        is always reserved for missing values. Must be no larger than 255.
    categorical_features : array-like of {bool, int, str} of shape (n_features) \
            or shape (n_categorical_features,), default=None
        Indicates the categorical features.

        - None : no feature will be considered categorical.
        - boolean array-like : boolean mask indicating categorical features.
        - integer array-like : integer indices indicating categorical
          features.
        - str array-like: names of categorical features (assuming the training
          data has feature names).

        For each categorical feature, there must be at most `max_bins` unique
        categories, and each categorical value must be less then `max_bins - 1`.
        Negative values for categorical features are treated as missing values.
        All categorical values are converted to floating point numbers.
        This means that categorical values of 1.0 and 1 are treated as
        the same category.
    monotonic_cst : array-like of int of shape (n_features) or dict, default=None
        Monotonic constraint to enforce on each feature are specified using the
        following integer values:

        - 1: monotonic increase
        - 0: no constraint
        - -1: monotonic decrease

        If a dict with str keys, map feature to monotonic constraints by name.
        If an array, the features are mapped to constraints by position. See
        :ref:`monotonic_cst_features_names` for a usage example.

        The constraints are only valid for binary classifications and hold
        over the probability of the positive class.
    interaction_cst : {"pairwise", "no_interactions"} or sequence of lists/tuples/sets \
            of int, default=None
        Specify interaction constraints, the sets of features which can
        interact with each other in child node splits.

        Each item specifies the set of feature indices that are allowed
        to interact with each other. If there are more features than
        specified in these constraints, they are treated as if they were
        specified as an additional set.

        The strings "pairwise" and "no_interactions" are shorthands for
        allowing only pairwise or no interactions, respectively.

        For instance, with 5 features in total, `interaction_cst=[{0, 1}]`
        is equivalent to `interaction_cst=[{0, 1}, {2, 3, 4}]`,
        and specifies that each branch of a tree will either only split
        on features 0 and 1 or only split on features 2, 3 and 4.
    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble. For results to be valid, the
        estimator should be re-trained on the same data only.
        See :term:`the Glossary <warm_start>`.
    early_stopping : 'auto' or bool, default='auto'
        If 'auto', early stopping is enabled if the sample size is larger than
        10000. If True, early stopping is enabled, otherwise early stopping is
        disabled.
    scoring : str or callable or None, default='loss'
        Scoring parameter to use for early stopping. It can be a single
        string (see :ref:`scoring_parameter`) or a callable (see
        :ref:`scoring`). If None, the estimator's default scorer is used. If
        ``scoring='loss'``, early stopping is checked w.r.t the loss value.
        Only used if early stopping is performed.
    validation_fraction : int or float or None, default=0.1
        Proportion (or absolute size) of training data to set aside as
        validation data for early stopping. If None, early stopping is done on
        the training data. Only used if early stopping is performed.
    n_iter_no_change : int, default=10
        Used to determine when to "early stop". The fitting process is
        stopped when none of the last ``n_iter_no_change`` scores are better
        than the ``n_iter_no_change - 1`` -th-to-last one, up to some
        tolerance. Only used if early stopping is performed.
    tol : float, default=1e-7
        The absolute tolerance to use when comparing scores during early
        stopping. The higher the tolerance, the more likely we are to early
        stop: higher tolerance means that it will be harder for subsequent
        iterations to be considered an improvement upon the reference score.
    verbose : int, default=0
        The verbosity level. If not zero, print some information about the
        fitting process.
    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the subsampling in the
        binning process, and the train/validation data split if early stopping
        is enabled.
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    estimators_ : list of HistGradientBoostingRegressor
        List containing the models for each quantile of interest
    corrections_ : dict of lists of floats
        the dictionary uses the cluster label as key, the values are arrays 
        of floats to be added to the quantile predictions to conformalize
    do_early_stopping_ : bool
        Indicates whether early stopping is used during training.
    n_iter_ : int
        The number of iterations as selected by early stopping, depending on
        the `early_stopping` parameter. Otherwise it corresponds to max_iter.
    n_trees_per_iteration_ : int
        The number of tree that are built at each iteration. For regressors,
        this is always 1.
    train_score_ : ndarray, shape (n_iter_+1,)
        The scores at each iteration on the training data. The first entry
        is the score of the ensemble before the first iteration. Scores are
        computed according to the ``scoring`` parameter. If ``scoring`` is
        not 'loss', scores are computed on a subset of at most 10 000
        samples. Empty if no early stopping.
    validation_score_ : ndarray, shape (n_iter_+1,)
        The scores at each iteration on the held-out validation data. The
        first entry is the score of the ensemble before the first iteration.
        Scores are computed according to the ``scoring`` parameter. Empty if
        no early stopping or if ``validation_fraction`` is None.
    is_categorical_ : ndarray, shape (n_features, ) or None
        Boolean mask for the categorical features. ``None`` if there are no
        categorical features.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    References
    --------
    [1] Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized quantile 
        regression. Advances in neural information processing systems, 32.
    [2] Sousa, M., Tomé, A. M., & Moreira, J. (2024). Improving conformalized 
        quantile regression through cluster-based feature relevance. Expert 
        Systems with Applications, 238, 122322.

    Examples
    --------
    >>> from sklearn.ensemble import HistGradientBoostingRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> est = HistGradientBoostingRegressor().fit(X, y)
    >>> est.score(X, y)
    0.92...
    """
    
    def __init__(
        self,
        quantiles=[0.5],
        n_samples=100,
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_bins=255,
        categorical_features=None,
        monotonic_cst=None,
        interaction_cst=None,
        warm_start=False,
        early_stopping="auto",
        scoring="loss",
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
        verbose=0,
        random_state=None,
        ):
        super(ConfGradientBoostingRegressor, self).__init__(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_bins=max_bins,
            monotonic_cst=monotonic_cst,
            interaction_cst=interaction_cst,
            categorical_features=categorical_features,
            early_stopping=early_stopping,
            warm_start=warm_start,
            scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )
        self.quantiles = quantiles
        self.n_samples = n_samples
        self.corrections = {}
        self.quantiles_ = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
        self.estimators_ = []
    
    
    def optimize(self, X, y, n_iter=10):
        """Optimize model parameters with Randomized Search Cross Validation
        Optimized parameters are:
            max_leaf_nodes : between 10 and 50
            max_depth : between 3 and 20
            max_iter : between 50 and 100
            learning_rate: between 0. and 1.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            Target values.

        n_iter : int, default=10
            Number of iterations to search for optimal parameters
            
        Returns
        -------
        self : object
            Estimator with optimized parameters.
        """
        
        params = self.get_params().copy()
        params.pop('quantiles')
        params['loss'] = 'quantile'
        params['quantile'] = 0.5
        estimator = HistGradientBoostingRegressor(**params)
        params_distributions = dict(
            max_leaf_nodes=randint(low=10, high=50),
            max_depth=randint(low=3, high=20),
            max_iter=randint(low=50, high=100),
            learning_rate=uniform()
        )
        optim_model = RandomizedSearchCV(
            estimator,
            param_distributions=params_distributions,
            n_jobs=-1,
            n_iter=n_iter,
            cv=KFold(n_splits=5, shuffle=True),
            verbose=0
        )
        optim_model.fit(X_train, y_train)
        estimator = optim_model.best_estimator_
        self.max_iter = estimator.get_params()['max_iter']
        self.learning_rate = estimator.get_params()['learning_rate']
        
        return self
    
    def _sample(self, q):
        return np.interp(np.random.uniform(size=self.n_samples), 
                         np.linspace(0,1,len(q)), 
                         q.sort_values())
    
    def fit(self, X, y, sample_weight=None):
        """Fit the gradient boosting models for the quantiles.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,) default=None
            Weights of training data.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        
        for q in self.quantiles_:
            params = self.get_params().copy()
            params.pop('quantiles')
            params['loss'] = 'quantile'
            params['quantile'] = q
            estimator = HistGradientBoostingRegressor(**params)
            estimator.fit(X, y, sample_weight)
            self.estimators_.append(estimator)
        
        return self
    
    
    def score(self, X, y, sample_weight=None):
        """Reurns the sum of pinball losses for the quantiles.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,) default=None
            Weights of training data.

        Returns
        -------
        score : float
        """
        score = 0.
        for i in range(len(self.quantiles_)):
            estimator = self.estimators_[i]
            score += mean_pinball_loss(y, 
                                       self.estimators_[i].predict(X), 
                                       alpha=self.quantiles_[i], 
                                       sample_weight=sample_weight)
        return score
    
    
    def calibrate(self, X, y, c=None):
        """Calibrate the conformalization corrections for each cluster
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            Target values.

        c : array-like, shape (n_samples,)
            The clusters labels for conformalization.
            
        Returns
        -------
        self : object
            Estimator with fitted calibration corrections.
        """
        
        if c is None:
            c = np.zeros(len(y), dtype=int)
        
        y_hat = self.predict(X)
        
        for l in np.unique(c):
            self.corrections[l] = []
            for i in range(len(self.quantiles)):
                error = y[c==l] - y_hat[c==l,i]
                self.corrections[l].append(np.quantile(error, self.quantiles[i], method="higher"))
        
        return self
    
        
    def predict(self, X):
        """Predict values for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples, n_quantiles)
            The predicted values.
        """
        
        q = []
        for e in self.estimators_:
            q.append(e.predict(X))
        q = np.stack(q, axis=1)
        q = pd.DataFrame(q)
        ys = q.apply(self._sample, axis=1, result_type='expand')
        
        return ys.quantile(self.quantiles, axis=1).values.T
    
    
    def conformalize(self, y, c=None):
        """Conformalize values for y.

        Parameters
        ----------
        y : array-like, shape (n_samples, n_quantiles)
            The predictions to be conformalized.
        c : array-like, shape (n_samples,)
            The clusters labels for conformalization.

        Returns
        -------
        yc : ndarray, shape (n_samples, n_quantiles)
            The conformalized predicitions.
        """
        
        if c is None:
            c = np.zeros(len(y), dtype=int)
        
        yc = y.copy()
        for l in np.unique(c):
            yc[c==l] = yc[c==l] + self.corrections[l]
        
        return yc
        