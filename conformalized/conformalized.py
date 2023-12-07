import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

class ConfGradientBoostingRegressor(GradientBoostingRegressor):
    """Conformalized Histogram-based Gradient Boosting Regression Tree.

    This estimator is based on
    :class:`GradientBoostingRegressor<sklearn.ensemble.GradientBoostingRegressor>`
    with additional methods for optimization of the parameters through cross 
    validation and conformalization of the prediction handling heteroscedasticity.

    This estimator builds an additive model in a forward stage-wise fashion; it
    allows for the optimization of arbitrary differentiable loss functions. In
    each stage a regression tree is fit on the negative gradient of the given
    loss function.
    
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
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        Values must be in the range `[0.0, inf)`.

    n_estimators : int, default=500
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        Values must be in the range `[1, inf)`.

    subsample : float, default=0.5
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.
        Values must be in the range `(0.0, 1.0]`.

    criterion : {'friedman_mse', 'squared_error'}, default='friedman_mse'
        The function to measure the quality of a split. Supported criteria are
        "friedman_mse" for the mean squared error with improvement score by
        Friedman, "squared_error" for mean squared error. The default value of
        "friedman_mse" is generally the best as it can provide a better
        approximation in some cases.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, values must be in the range `[2, inf)`.
        - If float, values must be in the range `(0.0, 1.0]` and `min_samples_split`
          will be `ceil(min_samples_split * n_samples)`.

    min_samples_leaf : int or float, default=0.01
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, values must be in the range `[1, inf)`.
        - If float, values must be in the range `(0.0, 1.0)` and `min_samples_leaf`
          will be `ceil(min_samples_leaf * n_samples)`.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
        Values must be in the range `[0.0, 0.5]`.

    max_depth : int or None, default=3
        Maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        If int, values must be in the range `[1, inf)`.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        Values must be in the range `[0.0, inf)`.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    init : estimator or 'zero', default=None
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide :term:`fit` and :term:`predict`. If 'zero', the
        initial raw predictions are set to zero. By default a
        ``DummyEstimator`` is used, predicting either the average target value
        (for loss='squared_error'), or a quantile for the other losses.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to each Tree estimator at each
        boosting iteration.
        In addition, it controls the random permutation of the features at
        each split (see Notes for more details).
        It also controls the random splitting of the training data to obtain a
        validation set if `n_iter_no_change` is not None.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    max_features : {'sqrt', 'log2'}, int or float, default='sqrt'
        The number of features to consider when looking for the best split:

        - If int, values must be in the range `[1, inf)`.
        - If float, values must be in the range `(0.0, 1.0]` and the features
          considered at each split will be `max(1, int(max_features * n_features_in_))`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.
        Values must be in the range `[0, inf)`.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        Values must be in the range `[2, inf)`.
        If None, then unlimited number of leaf nodes.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    validation_fraction : float, default=0.2
        The proportion of training data to set aside as validation set for
        early stopping. Values must be in the range `(0.0, 1.0)`.
        Only used if ``n_iter_no_change`` is set to an integer.

    n_iter_no_change : int, default=10
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.
        Values must be in the range `[1, inf)`.

    tol : float, default=1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.
        Values must be in the range `[0.0, inf)`.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.
        Values must be in the range `[0.0, inf)`.
        See :ref:`minimal_cost_complexity_pruning` for details.

    Attributes
    ----------
    estimators_ : list of HistGradientBoostingRegressor
        List containing the models for each quantile of interest
        
    corrections_ : dict of lists of floats
        the dictionary uses the cluster label as key, the values are arrays 
        of floats to be added to the quantile predictions to conformalize
        
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    oob_improvement_ : ndarray of shape (n_estimators,)
        The improvement in loss on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.
        Only available if ``subsample < 1.0``.

    oob_scores_ : ndarray of shape (n_estimators,)
        The full history of the loss values on the out-of-bag
        samples. Only available if `subsample < 1.0`.

    oob_score_ : float
        The last value of the loss on the out-of-bag samples. It is
        the same as `oob_scores_[-1]`. Only available if `subsample < 1.0`.

    train_score_ : ndarray of shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the loss of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the loss on the training data.

    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : ndarray of DecisionTreeRegressor of shape (n_estimators, 1)
        The collection of fitted sub-estimators.

    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    max_features_ : int
        The inferred value of max_features.

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
        *,
        quantiles=[0.5],
        learning_rate=0.1,
        n_estimators=500,
        subsample=0.5,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=0.01,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        random_state=None,
        max_features="sqrt",
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.2,
        n_iter_no_change=10,
        tol=1e-4,
        ccp_alpha=0.0,
    ):
        super().__init__(
            loss='quantile',
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            alpha=0.5,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
        )
        self.quantiles = quantiles
        self.corrections_ = {}
        self.estimators_ = []
    
    
    def _sample(self, q, size=100):
        return np.interp(np.random.uniform(size=size), 
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
        
        self.estimators_ = []
        for q in self.quantiles:
            params = self.get_params().copy()
            params.pop('quantiles')
            params['loss'] = 'quantile'
            params['alpha'] = q
            estimator = GradientBoostingRegressor(**params)
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
        for i in range(len(self.quantiles)):
            estimator = self.estimators_[i]
            score += mean_pinball_loss(y, 
                                       self.estimators_[i].predict(X), 
                                       alpha=self.quantiles[i], 
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
            self.corrections_[l] = []
            for i in range(len(self.quantiles)):
                error = y[c==l] - y_hat[c==l,i]
                self.corrections_[l].append(np.quantile(error, self.quantiles[i], method="higher"))
        
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
        
        y = []
        for e in self.estimators_:
            y.append(e.predict(X))
        return np.stack(y, axis=1)
    
    
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
            yc[c==l] = yc[c==l] + self.corrections_[l]
        
        return yc
        