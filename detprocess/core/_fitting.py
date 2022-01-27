import numpy as np
from scipy.optimize import curve_fit
import iminuit


__all__ = [
    "ext_max_llhd",
    "NormBackground",
]


def ext_max_llhd(x, func, guess, guess_err=None, limits=None):
    """
    Routine for finding the Extended Unbinned Maximum Likelihood of an
    inputted spectrum, giving an inputted (arbitrary) function.

    Parameters
    ----------
    x : array_like
        The energies of each inputted event.
    func : FunctionType
        The negative log-likelihood for the normalized PDF, see Notes.
    guess : array_like
        Guesses for the true values of each parameter.
    guess_err : array_like, optional
        Guess for the errors of each parameter. Default is to
        simply use the guesses.
    limits : array_like, optional
        The limits to set on each parameter. Default is to set no
        limits. Should be of form:
            [(lower0, upper0), (lower1, upper1), ...]

    Returns
    -------
    m : iminuit.Minuit
        The Minuit object that contains all information on the fit,
        after the MINUIT algorithm has completed.

    Notes
    -----
    For a normalized PDF of form f(x, p) / Norm(p), where p is a vector
    of the fit parameters, the negative-log likelihood for the Extended
    Unbinned Maximum Likelihood method is:

        -log(L) = Norm(p) - sum(log(f(x, p)))

    """

    if guess_err is None:
        guess_err = [g for ii, g in enumerate(guess)]

    if limits is None:
        limits = [(-np.inf, np.inf) for ii in range(len(guess))]

    m = iminuit.Minuit(
        lambda p: func(x, p),
        guess,
        name=[f'p{ii}' for ii in range(len(guess))],
    )
    m.limits = limits
    m.errors = guess_err
    m.errordef = 1

    m.migrad()
    m.hesse()

    return m


class NormBackground(object):
    """
    Class for calculating a normalized spectrum from specified
    background shapes.

    """

    def __init__(self, lwrbnd, uprbnd, flatbkgd=True, nexpbkgd=0,
                 ngaussbkgd=0):
        """
        Initalization of the NormBackground class.

        Parameters
        ----------
        lwrbnd : float
            The lower bound of the background spectra, in energy.
        uprbnd : float
            The upper bound of the background spectra, in energy.
        flatbkgd : bool, optional
            If True, then the background spectrum will have a flat
            background component. If False, there will not be one.
            Default is True.
        nexpbkgd : int, optional
            The number of exponential spectra in the background
            spectrum. Default is 0.
        ngaussbkgd : int, optional
            The number of Gaussian spectra in the background spectrum.
            Default is 0.

        """

        self._lwrbnd = lwrbnd
        self._uprbnd = uprbnd
        self._bool_flatbkgd = flatbkgd
        self._nexpbkgd = nexpbkgd
        self._ngaussbkgd = ngaussbkgd

        self._nparams = flatbkgd + nexpbkgd * 2 + ngaussbkgd * 3

    @staticmethod
    def _flatbkgd(x, *p):
        """Hidden method to calculate a flat spectrum."""

        if np.isscalar(x):
            return p[0]

        return p[0] * np.ones(len(x))

    @staticmethod
    def _expbkgd(x, *p):
        """Hidden method to calculate an exponential spectrum."""

        return p[0] * np.exp(-x / p[1])

    @staticmethod
    def _gaussbkgd(x, *p):
        """Hidden method to calculate a Gaussian spectrum."""

        return p[0] * np.exp(-(x - p[1])**2 / (2 * p[2]**2))

    def background(self, x, p):
        """
        Method for calculating the differential background (in units of
        1 / [energy]) given the inputted parameters and initialized
        background shape.

        Parameters
        ----------
        x : ndarray
            The energies at which the background will be calculated.
        p : array_like
            The parameters that determine the shape of each component
            of the background. See Notes for order of parameters.

        Returns
        -------
        output : ndarray
            The differential background spectrum at each `x` value,
            given the inputted shape parameters `p`. Units of
            1 / [energy].

        Notes
        -----
        The order of parameters should be

            1) The flat background rate (if there is one, otherwise
                skip)
            2) The exponential shape parameters for each exponential
                background (if nonzero):
                    (amplitude, exponential coefficient)
            3) The Gaussian shape parameters for each Gaussian
                background (if nonzero):
                    (ampltiude, mean, standard deviation)

        """

        if len(p) != self._nparams:
            raise ValueError(
                'Length of p does not match expected number of parameters'
            )

        output = np.zeros(len(x))
        ii = 0

        if self._bool_flatbkgd:
            output += self._flatbkgd(x, *(p[ii], ))
            ii += 1

        for jj in range(self._nexpbkgd):
            output += self._expbkgd(x, *(p[ii], p[ii + 1]))
            ii += 2

        for jj in range(self._ngaussbkgd):
            output += self._gaussbkgd(x, *(p[ii], p[ii + 1], p[ii + 2]))
            ii += 3

        return output

    def _normalization(self, p):
        """
        Hidden method for calculating the normalization of the
        background spectrum.

        """

        norm = 0
        ii = 0

        if self._bool_flatbkgd:
            norm += self._flatbkgd(
                0, *(p[ii], ),
            ) * (self._uprbnd - self._lwrbnd)
            ii += 1

        for jj in range(self._nexpbkgd):
            norm += p[ii + 1] * (
                self._expbkgd(
                    self._lwrbnd, *(p[ii], p[ii + 1]),
                ) - self._expbkgd(
                    self._uprbnd, *(p[ii], p[ii + 1]),
                )
            )
            ii += 2

        for jj in range(self._ngaussbkgd):
            norm += p[ii] * np.sqrt(2 * np.pi * p[ii + 2]**2)
            ii += 3

        return norm

    def neglogllhd(self, x, p):
        """
        Method for calculating the negative log-likelihood for use with
        an extended maximum likelihood method, e.g.
        `rqpy.ext_max_llhd`.

        Parameters
        ----------
        x : ndarray
            The energies at which the background will be calculated.
        p : array_like
            The parameters that determine the shape of each component
            of the background. See Notes for order of parameters.

        Returns
        -------
        out : float
            The extended maximum likelihood for inputted spectrum
            parameters.

        Notes
        -----
        The order of parameters should be

            1) The flat background rate (if there is one, otherwise
                skip)
            2) The exponential shape parameters for each exponential
                background (if nonzero):
                    (amplitude, exponential coefficient)
            3) The Gaussian shape parameters for each Gaussian
                background (if nonzero):
                    (ampltiude, mean, standard deviation)

        """

        return -sum(np.log(self.background(x, p))) + self._normalization(p)

