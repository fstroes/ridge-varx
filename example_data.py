import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from VAR_fs import RidgeVARX

def simulate_data(random_seed):
    # set random seed for reproducibility
    np.random.seed(random_seed)

    '''SECTION 1 Simulate Data'''
    '''SECTION 1.1 Setting the length of the time series and the parameters for the simulation'''
    df = pd.DataFrame(index=pd.period_range('06-01-2018', '01-01-2021', freq='W-SUN'))
    T = len(df)
    n_out_sample = 52

    # define how many ar1 time series to create
    n_ar1s = 8
    # set autoregressive parameters the ar1 processes
    phis = list(np.random.uniform(.8, .9, size=n_ar1s))

    # model parameters that define the relationship between explanatory variables (at different lags) and the outcome
    betas = list(np.random.uniform(-5, 5, size=5))

    '''SECTION 1.2 simulate indpendent series from ar1 processes'''

    # Definition of the AR(1) recursion:
    def ar1(phi, T):
        series = pd.Series(np.random.normal(0, 1, T))  # start out with the innovations only
        for idx, item in series.iloc[:-1].iteritems():  # run the AR1 recursion
            series[idx + 1] += item * phi
        return series.values


    # creating the time series based on the ar1 processes
    ar1_names = [f'y{x}' for x in range(1, 4)]
    for series_name, phi in zip(ar1_names, phis):
        df[series_name] = ar1(phi, T)

    '''SECTION 1.3 simulate exogenous variables'''
    # two exogenous variables are added one is related, the other is unrelated to the outcome
    df[['related_exogenous', 'unrelated_exogenous']] = np.random.normal(0, 3, [T, 2])

    '''SECTION 1.4 Simulate an outcome that is a combination of the explanatory variables'''
    noise_scale = 1
    df['innovs0'] = np.random.normal(0, noise_scale, len(df))

    # Produce y0 as a linear combination of (lags of) the other time series
    df['y0'] = \
        betas[0] + \
        betas[1] * df['y1'].shift(1) + \
        betas[2] * df['y2'].shift(2) + \
        betas[3] * df['y3'].shift(3) + \
        betas[4] * df['related_exogenous'] + \
        df['innovs0']


    '''SECTION 1.5 define a train-test split'''
    observed = df.iloc[:, ~df.columns.str.contains('innovs')]
    in_sample, out_sample = observed.iloc[:-n_out_sample].dropna(), observed.tail(n_out_sample)

    return [in_sample, out_sample, phis, betas, ar1_names]

def rescale_params(scales_df, mod_params):
    mod_params = mod_params.copy()

    scale_original_cols = mod_params \
                              .rename(
        columns=lambda idx: idx.split('_explanatory')[0] if 'explanatory' in idx else idx).T.loc[:, []]
    scale_original_cols['scale_original'] = np.nan
    scale_original_cols.update(scales_df)
    scale_original_cols = scale_original_cols.set_index(mod_params.columns)

    scale_original_rows = scales_df
    scale_original_rows = scale_original_rows[~scale_original_rows.index.str.contains('exogenous')].set_index(
        mod_params.index)

    parameters_rescaled = mod_params / scale_original_cols[
        'scale_original']  # /scale_original_rows #/scale_original_rows['scale_original'] #
    parameters_rescaled = (parameters_rescaled.T * scale_original_rows['scale_original']).T
    return parameters_rescaled

def plot_ridge_grid(mod, fit_kwargs, ridge_grid):
    """
    :param mod:
    :param fit_kwargs:
    :param ridge_grid:
    :return:
    """

    penalty_val = []
    aic_val = []
    for x in ridge_grid:
        diff_AIC = mod\
            .fit(**fit_kwargs, ridge_penalty=x).diff_AIC  # the default behavior is to standardize the data but we did that already

        penalty_val.append(x)
        aic_val.append(diff_AIC)

    opt_ridge_penalty = penalty_val[aic_val.index(min(aic_val))]
    plt.plot(penalty_val, aic_val)
    plt.xlabel('$\lambda_i$')
    plt.ylabel('$\Delta AIC$')
    plt.title('Variable part of AIC for different values of $\lambda$')
    plt.savefig('images/ridge.png', dpi='figure',bbox_inches='tight')
    plt.show()

    return opt_ridge_penalty

def plot_out_of_sample(forecast_MLS, forecast_Ridge, out_sample_transformed):
    n_out_of_sample = len(out_sample_transformed)
    oos_MSE_MLS = ((forecast_MLS['y0'].iloc[-n_out_of_sample:] - out_sample_transformed['y0']) ** 2).mean()
    oos_MSE_Ridge = ((forecast_Ridge['y0'].iloc[-n_out_of_sample:] - out_sample_transformed['y0']) ** 2).mean()

    plt.clf()
    out_sample_transformed['y0'].plot(color='k')
    forecast_MLS['y0'].iloc[-n_out_of_sample:].plot(color='cyan', linestyle='--', ax=plt.gca())
    forecast_Ridge['y0'].iloc[-n_out_of_sample:].plot(color='purple', linestyle='--', ax=plt.gca())
    plt.title('Out of sample forecast')
    plt.legend(['actual', f'Forecast MLS (MSE={oos_MSE_MLS:.2f})', f'Forecast Ridge (MSE={oos_MSE_Ridge:.2f})'])
    plt.savefig('images/oos.png', dpi='figure')
    plt.show()

def plot_part_of_sample(in_sample):
    displ_n_plot = 100
    in_sample.iloc[-displ_n_plot:].plot(color='k', linewidth=.5)
    plt.title('Time series in the dataset')
    plt.legend().remove()
    plt.ylabel('$y_i$')
    plt.savefig('images/series.png', dpi='figure')
    plt.show()

def plot_true_vs_fit(rescaled_MLS, rescaled_Ridge, betas, phis, ar1_names):
    # We will contain the true parameters in the same output format as the parameters of the fitted models:
    true_params = rescaled_MLS.copy().stack()
    true_params[:] = 0  # all parameters in our simulated data are zero,
    # except for the ones that we specified

    # Fill the true parameters df with the betas that we set
    true_params[('predict_y0', 'y1_explanatory_lag1')] = betas[1]
    true_params[('predict_y0', 'y2_explanatory_lag2')] = betas[2]
    true_params[('predict_y0', 'y3_explanatory_lag3')] = betas[3]
    true_params[('predict_y0', 'related_exogenous')] = betas[4]

    # add the phis for the autoregressive processes too
    for ar1_name, phi in zip(ar1_names, phis):
        true_params[(f'predict_{ar1_name}', f'{ar1_name}_explanatory_lag1')] = phi

    drop_cons = lambda df: df[~df.index.get_level_values(1).str.contains('constant')]
    sort_by_abs = lambda df: df.reindex(df.abs().sort_values(ascending=False).index)

    true_params = sort_by_abs(true_params)
    colors = ['k', 'blue', 'magenta']
    fig, ax = plt.subplots()
    for i, params in enumerate([true_params, rescaled_MLS.stack(), rescaled_Ridge.stack()]):
        params.loc[true_params.index].plot(ax=ax, color=colors[i])
    ax.axes.xaxis.set_visible(True)
    ax.axes.xaxis.set_ticks([])
    plt.title('True versus fitted parameters values')
    plt.xlabel('Parameters sorted by absolute value')
    plt.ylabel('Parameter estimate')
    plt.legend(['True', 'MLS', 'Ridge'])
    plt.savefig('images/params.png', dpi='figure')
    plt.show()

    return None