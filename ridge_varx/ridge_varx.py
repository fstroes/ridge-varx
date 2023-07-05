import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings
import scipy
import scipy.sparse.linalg

# function that converts the Y to a Z matrix using numpy only (this way we can cythonize it later)
def y_to_z(Y, X, lags=(1,), deterministic_terms=('constant',), future_col=False):
    k_det = len(deterministic_terms)
    k_endog = len(lags)
    p_order = np.max(lags)

    # add NaNs so the t+1 row wont be lost in rolling
    Y = np.concatenate([Y.copy(), np.ones([1, Y.shape[1]])*np.nan])
    YT = Y.T
    X = X.copy()
    XT = X.T

    for i in range(k_det):
        if deterministic_terms[i] == 'constant':
            Z = np.ones([1, YT.shape[1]])


    for i in range(k_endog):
        # create lagged values
        lag = lags[i]
        lagged = np.roll(YT, lag)
        lagged[:, :lag] = np.ones(lagged[:, :lag].shape) * np.nan

        # create or append to Z depending on whether it exists already
        if i == 0 and not len(deterministic_terms):
            Z = lagged
        else:
            Z = np.concatenate([Z, lagged])

    # add the exogenous variables if exogenous is provided
    Z = np.concatenate([Z, XT]) if len(XT) else Z

    # remove columns with NaNs
    Z = Z[:, ~np.isnan(Z).any(axis=0)]

    #Z = Z[:, 2:] #align with observed Y
    if future_col:
        return Z
    else:
        return Z[:, :len(Y)]


def vec(A):
    """Vectorizes (in the mathematical sense) a matrix A -> vec(A)"""
    return(np.atleast_2d(A.flatten('F')).T)


# Here are some of the reformulations of the formula's
def cg_solver_for_matrix(A, B, asym=False, solver_kwargs=None):
    """ The cg solver handles vectors only for the outcome, we can rewrite any equation that outputs a vector 'b' given a
    matrix 'a' and a vector 'x' as an equation that outputas a matrix 'B' given a matrix 'X' instead and vice versa."""

    solver_kwargs = dict() if solver_kwargs is None else solver_kwargs

    b = vec(B)
    k = int(b.shape[0]/A.shape[0])
    m = A.shape[1]
    n = B.shape[1]
    A = np.kron(np.eye(k), A)
    x = scipy.sparse.linalg.cg(A, b, **solver_kwargs) if not asym else scipy.sparse.linalg.bicg(A, b, **solver_kwargs)
    X = x[0].reshape(m, n, order='F').T
    return X


def cg_solver_params(Z, Y, ridge_penalty, solver_kwargs=None):
    A = np.kron(np.eye(Y.shape[0]), Z.dot(Z.T) + ridge_penalty*np.eye(Z.shape[0]))

    solver_kwargs = dict() if solver_kwargs is None else solver_kwargs

    if 'M' not in solver_kwargs.keys(): # Use the approximate inverse as a starting point if no other is provided
        solver_kwargs.update(dict(M=np.linalg.inv(A)))

    b = vec(Z.dot(Y.T))
    x = scipy.sparse.linalg.cg(A, b, **solver_kwargs)
    B_hat = x[0].reshape(Z.shape[0], Y.shape[0], order='F').T
    return B_hat

def calc_C(Z, ridge_penalty, T, k):
    """
    :param Z: the matrix
    :param ridge_penalty:
    :return: (ZZ^T)^{-1} calculated with the smallest matrix inversion possible
    """

    ridge_penalty = ridge_penalty if ridge_penalty is not None else 0

    if k <= T:
        C = np.linalg.inv(Z.dot(Z.T) + ridge_penalty * np.eye(k))

    # if K_per_eq > n_stat_units invert the smaller TXT matrix using Woodbury identity
    else:
        C = (1 / ridge_penalty) * np.eye(k) \
            - (1 / (ridge_penalty ** 2)) * Z.values \
                .dot(
            np.linalg.inv(np.eye(T) + (1 / ridge_penalty) * Z.T.values.dot(Z.values))
        ).dot(Z.T.values)

    return C


# helper function that produces a new forecast row of all variables
def VAR_1step(Y, X, B, lags=(), deterministic_terms=()):
    Z = y_to_z(Y, X, lags, deterministic_terms, future_col=True)
    Ypred = B.dot(Z)
    last_row = Ypred[:, -1]
    return last_row


class RidgeVARX:
    def __init__(self, endog, exog=None):
        self.endog = endog.copy()

        self.exog = exog.copy() if exog is not None else exog

        _nan_str = 'Train_data can not contain NaN values!'

        if exog is not None:
            if not len(endog) == len(endog.dropna()):
                raise Exception(_nan_str)
            if not len(exog) == len(exog.dropna()):
                raise Exception(_nan_str)

            _not_alligned_str = 'data and exogenous are not aligned, exogenous can contain future values but all\
                                 values present in "data" must also be present in the "df_ex"'

            # if not all indices in the data are also in the df_ex..
            if not endog.index.isin(exog.index).all():
                raise ValueError(_not_alligned_str)

            # if the samples don't start at the same time
            if exog.index[0] != endog.index[0]:
                raise ValueError(_not_alligned_str)


            # if df_ex is not at least one period longer than data
            if len(exog) - len(endog) == 0:
                self.exog.loc[self.exog.iloc[-1].name + 1, :] = np.nan  # this is needed for alignment in the
                                                                            # y to z function

    class ModelFit:
        """
        A trained model object that can be used for prediction
        """
        def __init__(self, endog, exog, lags, deterministic_terms, ridge_penalty, standardize, solver_kwargs):

            if deterministic_terms is None:
                deterministic_terms = tuple()

            if standardize is None:
                standardize = True if ridge_penalty is not None else False

            self.standardize = standardize
            #  allow lags to be specified as p_max
            if type(lags) == int:
                lags = list(range(1, lags+1))

            self.ridge_penalty = ridge_penalty

            self.lags = lags
            deterministic_terms = list(deterministic_terms)
            self.deterministic_terms = deterministic_terms
            self.endog = endog
            self.exog = exog

            if standardize:
                # normalize observed vectors of endogenous and exogenous
                endog_scales = endog.var().apply(np.sqrt)
                endog_locations = endog.mean()
                endog_stand = (endog - endog_locations) / endog_scales

                self.endog_scales, self.endog_locations, self.endog_stand = endog_scales, endog_locations, endog_stand

                if exog is not None:
                    exog_scales = exog.var().apply(np.sqrt)
                    exog_locations = exog.mean()
                    exog_stand = (exog - exog_locations) / exog_scales

                    self.exog_scales, self.exog_locations, self.exog_stand = exog_scales, exog_locations, exog_stand

                # remove constant term from model if present (replaced by addition of endogenous variable location)
                if 'constant' in deterministic_terms:
                    deterministic_terms.remove('constant') # will also remove item from list stored in self


            self.k = endog.shape[1]  # number of endogenous variables
            self.m = exog.shape[1] if exog is not None else 0  # number of exogenous variables
            self.n_p = len(lags)  # number of lags of endogenous
            self.k_det = len(deterministic_terms)  # number of deterministic terms

            self.K = (self.n_p * self.k + self.m + self.k_det) * self.k  # total number of model parameters
            self.K_per_eq = int(self.K / self.k)

            for component in deterministic_terms:
                if component in ['constant']:
                    pass
                else:
                    raise Exception('Deterministic component not supported')


            # produce column names
            column_names_base = [f'{x}_explanatory' for x in
                                 endog.iloc[:, ~endog.columns.isin(deterministic_terms)].columns]
            columns_out_params = list(deterministic_terms)

            # create names for lagged values
            for lag in lags:
                columns_out_params += [f'{x}_lag{lag}' for x in column_names_base]

            if exog is not None:
                columns_out_params += list(exog.columns)

            if standardize:
                Z = pd.DataFrame(
                    data=y_to_z(endog_stand.values.astype(np.float64),
                                exog_stand.loc[:endog.index[-1] + 1].values.astype(np.float64) if exog is not None else np.array([]),
                                lags,
                                deterministic_terms,
                                future_col=False))
            else:
                Z = pd.DataFrame(
                    data=y_to_z(endog.values.astype(np.float64),
                                exog.loc[:endog.index[-1] + 1].values.astype(np.float64) if exog is not None else np.array([]),
                                lags,
                                deterministic_terms,
                                future_col=False))

            Z.index = columns_out_params

            # Only use the observations that we can link to lagged values
            n_stat_units = Z.shape[1]
            self.n_stat_units = n_stat_units

            if n_stat_units <= self.K_per_eq and (ridge_penalty is None or ridge_penalty == 0):
                raise Exception(f'Problem is ill posed without regularization.\
                 Available data points after taking lags ({n_stat_units})\
                 < total number of parameters per equation ({self.K_per_eq}) . Set ridge penalty > 0 to fit this model')

            self.full_rank_design_matrix = np.linalg.matrix_rank(Z.values.T, tol=1e-30) == Z.T.shape[1]

            if not self.full_rank_design_matrix:
                warnings.warn('Near perfect multicolinearity detected in design matrix,'
                              'OLS without penalty is not advised')

            if standardize:
                Y = endog_stand.T.iloc[:, (-n_stat_units):]
            else:
                Y = endog.T.iloc[:, (-n_stat_units):]

            # model estimation using multivariate least squares
            if ridge_penalty is None:
                B = cg_solver_params(Z.values, Y.values, ridge_penalty=0, solver_kwargs=solver_kwargs)

            # model estimation using ridge if ridge_penalty provided
            elif ridge_penalty is not None:
                B = cg_solver_params(Z.values, Y.values, ridge_penalty, solver_kwargs=solver_kwargs)

            C = calc_C(Z.values, ridge_penalty, self.n_stat_units, self.K_per_eq)
            H = Z.T.values.dot(C).dot(Z.values)

            # alternative for large K: https://arxiv.org/pdf/1509.09169;Lecture (page 21)

            self.g_df = np.trace(H)
            self.H = H
            self.C = C


            columns = columns_out_params
            row_indices = [f'predict_{col}' for col in list(endog.columns)]
            self.params = pd.DataFrame(data=B, columns=columns_out_params, index=row_indices)

            error_var_unadjusted = (1/self.n_stat_units) * (Y.values - B.dot(Z.values)).dot((Y.values - B.dot(Z.values)).T)

            if ridge_penalty is None or ridge_penalty == 0:
                df = self.K_per_eq
            else:
                df = self.g_df

            error_var = n_stat_units/(n_stat_units - df) * error_var_unadjusted

            self.error_var, self.error_var_unadjusted = error_var, error_var_unadjusted

            # estimation variance and p_values
            W_Ridge = C.dot(Z).dot(Z.T)

            estimation_cov = (np.kron(W_Ridge.dot(C).dot(W_Ridge.T), error_var))
            estimation_var = np.reshape(np.diag(estimation_cov), B.shape, 'F')

            self.estimation_cov = estimation_cov

            z_vals = self.params.values / np.sqrt(estimation_var)
            self.z_vals = pd.DataFrame(data=z_vals, columns=self.params.columns, index=self.params.index)

            p_val = lambda x: 1 - norm.cdf(abs(x))
            self.p_vals = self.z_vals.apply(p_val)

            signal_df = Y.copy().T

            signal_df[:] = B.dot(Z).T

            if standardize:
                signal_df = (signal_df * endog_scales) + endog_locations


            signal_df = signal_df.rename(columns=lambda x: f'{x}_predicted')

            # store some variables
            self.signal = signal_df
            self.Y, self.Z, self.B = Y, Z, B

            self.diff_AIC = np.log(np.linalg.det(self.error_var_unadjusted)) + 2 * df * self.k / self.n_stat_units

        def forecast(self, h, exog_future=None):
            """Forecast h-steps ahead"""
            endog = self.endog.copy() if not self.standardize is None else self.endog_stand.copy()

            if self.exog is not None:
                exog_future = exog_future if not self.standardize else (exog_future - self.exog_locations)\
                                                                             / self.exog_scales

                required_period_range = [endog.index[-1] + j for j in range(1, h+1)]
                missing_periods_exog = [x for x in required_period_range if x not in exog_future.index]

                if exog_future is None:
                    raise Exception(f'Exogenous variable values not supplied while exogenous component in model.')


                if len(missing_periods_exog):
                    raise Exception(f'missing exogenous variable endog for {missing_periods_exog}')

                df_ex = pd.concat([self.exog.dropna(), exog_future]) # drop nans if they were added for alignment

                if exog_future.index[-1] < endog.index[-1]:
                    raise Exception('Not enough values for the exogenous to make this forecast')

            # create the dynamic forecast in a for loop
            for j in range(1, h+1):
                endog = endog.append(pd.Series(data=[np.nan] * endog.shape[1], index=endog.columns, name=endog.index[-1]+1))


            for idx, row in endog.iloc[-h:, :].iterrows():
                data_in = endog.loc[:idx-1, :]
                prediction = VAR_1step(data_in.values,
                                       df_ex.loc[data_in.index[0]:data_in.index[-1] + 1].values
                                       if self.exog is not None else np.array([]), self.B,
                                       self.lags, self.deterministic_terms)
                endog.loc[idx, :] = prediction

            return (endog * self.endog_scales + self.endog_locations) if self.standardize else endog


    # the fit function that returns the fit object
    def fit(self, lags, deterministic_terms=('constant',), ridge_penalty=None, standardize=None, solver_kwargs=None):
        return self.ModelFit(self.endog, self.exog, lags, deterministic_terms, ridge_penalty, standardize, solver_kwargs)
