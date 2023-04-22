# MultiGaussianFit

Fit data supposed to be a sum of some gaussian distributions, i.e. gamma ray spectrum.

## Usage

See example at the bottom of the code. The `function multi_gaussian_fit` function returns a `FitResult` object, in which:
- `FitResult.coefs` column 0 is amplitudes, column 1 is $\mu$, and column 2 is $\sigma$.
- `FitResult.uncertain` reflects the quality of fitting. If it is too large, a warning will be thrown.
- `FitResult.func()` will return a function f(x) which returns the fitted function.

## Details

Data `x, y` will be rescaled to [0,1]x[0,1] for easier fitting, and the estimated $\mu$ is calculated first using **weighted KMeans clustering**. Then `scipy.curve_fit` will be employed to fit the curve.
