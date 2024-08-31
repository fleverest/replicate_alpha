import numpy as np

from testsmart.nnm import AlphaMart, ShrinkTrunc


def shrink_trunc(
    x: np.array,
    N: int,
    mu: float = 1 / 2,
    nu: float = 1 - np.finfo(float).eps,
    u: float = 1,
    c: float = 1 / 2,
    d: float = 100,
) -> np.array:
    r"""
    apply the shrinkage and truncation estimator to an array

    sample mean is shrunk towards nu, with relative weight d times the weight
    of a single observation.
    estimate is truncated above at u-u*eps and below at mu_j+e_j(c,j)

    S_1 = 0
    S_j = \sum_{i=1}^{j-1} x_i, j > 1
    m_j = (N*mu-S_j)/(N-j+1) if np.isfinite(N) else mu
    e_j = c/sqrt(d+j-1)
    eta_j =  ( (d*nu + S_j)/(d+j-1) \vee (m_j+e_j) ) \wedge u*(1-eps)

    Parameters
    ----------
    x : np.array
        input data
    mu : float in (0, 1)
        hypothesized population mean
    eta : float in (t, 1)
        initial alternative hypothethesized value for the population mean
    c : positive float
        scale factor for allowing the estimated mean to approach t from above
    d : positive float
        relative weight of nu compared to an observation, in updating the
        alternative for each term
    """
    S = np.insert(np.cumsum(x), 0, 0)[0:-1]  # 0, x_1, x_1+x_2, ...,
    j = np.arange(1, len(x) + 1)  # 1, 2, 3, ..., len(x)
    m = (
        (N * mu - S) / (N - j + 1) if np.isfinite(N) else mu
    )  # mean of population after (j-1)st draw, if null is true
    return np.minimum(
        u * (1 - np.finfo(float).eps),
        np.maximum((d * nu + S) / (d + j - 1), m + c / np.sqrt(d + j - 1)),
    )


def alpha_mart(
    x: np.array,
    N: int,
    mu: float = 1 / 2,
    eta: float = 1 - np.finfo(float).eps,
    u: float = 1,
    estim: callable = shrink_trunc,
) -> np.array:
    r"""
    Finds the ALPHA martingale for the hypothesis that the population
    mean is less than or equal to t using a martingale method,
    for a population of size N, based on a series of draws x.

    The draws must be in random order, or the sequence is not a martingale
    under the null

    If N is finite, assumes the sample is drawn without replacement
    If N is infinite, assumes the sample is with replacement

    Parameters
    ----------
    x : list corresponding to the data
    N : int
        population size for sampling without replacement, or np.infinity for
        sampling with replacement
    mu : float in (0,1)
        hypothesized fraction of ones in the population
    eta : float in (t,1)
        alternative hypothesized population mean
    estim : callable
        estim(x, N, mu, eta, u) -> np.array of length len(x), the sequence of
        values of eta_j for ALPHA

    Returns
    -------
    terms : array
        sequence of terms that would be a nonnegative supermartingale under
        the null
    """
    S = np.insert(np.cumsum(x), 0, 0)[0:-1]  # 0, x_1, x_1+x_2, ...,
    j = np.arange(1, len(x) + 1)  # 1, 2, 3, ..., len(x)
    m = (
        (N * mu - S) / (N - j + 1) if np.isfinite(N) else mu
    )  # mean of population after (j-1)st draw, if null is true
    etaj = estim(x, N, mu, eta, u)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.cumprod((x * etaj / m + (u - x) * (u - etaj) / (u - m)) / u)
    terms[m < 0] = np.inf
    return terms


def old_alpha(x, eta, d):
    c = 0.5 * (eta - 1 / 2)
    return alpha_mart(
        x,
        np.inf,
        mu=1 / 2,
        eta=eta,
        u=1,
        estim=lambda x, N, mu, eta, u: shrink_trunc(x, N, mu, eta, 1, c=c, d=d),
    )


def new_alpha(x, eta, d):
    test = AlphaMart(estim=ShrinkTrunc(eta0=eta, d=d))
    for xi in x:
        _ = test.update([xi])
        if test.stopped:
            break
    return np.array(test.e_process)
