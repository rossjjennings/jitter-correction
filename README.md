# Jitter Correction
This package implements several algorithms for using pulse profile shapes to correct for jitter noise in pulsar timing. Details of these methods are described in an associated paper (submitted to ApJ; preprint available at [arXiv:2605.21750](https://arxiv.org/abs/2605.21750)).
## Installation
This is an installable Python package, but is not (currently) available on PyPI. It either be used with the bundled `pixi` environment or installed directly using `pip`.
### `pixi`
It is easiest to use this package with the included `pixi` environment specification, which will create a new virtual environment containing exact pinned versions of all dependencies:
```
curl -fsSL https://pixi.sh/install.sh | sh
git clone https://github.com/rossjjennings/jitter-correction
cd jitter-correction
pixi install
```
The environment can then be activated using `pixi shell`. This creates a subshell that can be exited, e.g., with `Ctrl+D` or `exit`.

To use the `pixi` environment with an existing Jupyter notebook or JupyterLab installation, run
```
pixi run install-kernel
```
This should install a Jupyter kernel named `jitter-correction`. You might have to reload the Jupyter session for the new kernel to appear.
### Manual (`pip`)
Alternatively, you can install `jitter-correction` in your own Python virtual environment by cloning this repository and installing from the local copy:
```
git clone https://github.com/rossjjennings/jitter-correction
cd jitter-correction
pip install .
```
or directly from GitHub, without making a local clone of the repository:
```
pip install git+https://github.com/rossjjennings/jitter-correction
```
With this install method, `pip` should locate and install compatible versions of the dependencies automatically, but they may not be the same versions used in the `pixi` environment.
## Use
The `jitter-correction` package is meant to be used as a library in Python scripts and Jupyter notebooks. There are two main APIs: a high-level Analysis API and a lower-level Estimator API. An example of using the high-level Analysis API is the following, which, up to the choice of random seed, replicates the first panel of Figure 4 in the paper:

```python
import numpy as np
import matplotlib.pyplot as plt
import jitter_correction as jc

seps = np.linspace(0, 0.10, 33)
results = []
for i, sep in enumerate(seps):
    spec = jc.PulseSpec.new(
        amplitude=[1.0, 1.0],
        loc=[-sep/2, sep/2],
        fwhm=[0.05, 0.05],
        fj=[0.0, 0.0],
        modindex=[1.0, 1.0],
    )
    profile_model = jc.ProfileModel(
        spec,
        n_profiles=512,
        npprof=1000,
        n_bins=2048,
        snr=1000,
        ampl_dist='lognorm',
        drift_bins=50,
    )
    result = jc.analysis.run_analyses(
        profile_model,
        {
            'tm': jc.analysis.TemplateMatchingAnalysis(),
            'pcr': jc.analysis.PCRegressionAnalysis(n_pcs=1),
            'pcm': jc.analysis.PCMatchingAnalysis(n_pcs=1),
            'pcb': jc.analysis.PCBayesianAnalysis(n_pcs=1),
            'sr': jc.analysis.SkewnessRegressionAnalysis(),
        },
    )
    results.append(result)
    print(f"\rCompleted separation {i+1} of {len(seps)}", end="")
print()

dtoas = {}
for analysis in ['tm', 'pcr', 'pcm', 'pcb', 'sr']:
    dts = []
    for i in range(len(results)):
        res = results[i]
        toa = res[analysis].toa_results.toa
        dt = toa - res.profile_model.shifts
        dt /= res.profile_model.n_bins
        dts.append(dt)
    dts = np.array(dts)
    dtoas[analysis] = dts

fig, ax = plt.subplots()
ax.plot(seps, np.std(dtoas['tm'], axis=1), color='C0', alpha=0.5,
	marker='.', label='Template matching')
ax.plot(seps, np.std(dtoas['sr'], axis=1), color='C2', alpha=0.5,
	marker='.', label='Skewness regression')
ax.plot(seps, np.std(dtoas['pcr'], axis=1), color='C4', alpha=0.5,
	marker='.', label='PC regression')
ax.plot(seps, np.std(dtoas['pcm'], axis=1), color='C1', alpha=0.5,
	marker='.', label='PC matching (no prior)')
ax.plot(seps, np.std(dtoas['pcb'], axis=1), color='C3', alpha=0.5
	marker='.', label='PC matching (with prior)')
ax.set_xlabel('Separation (cycles)')
ax.set_ylabel('ΔTOA (cycles)')
ax.set_yscale('log')
fig.legend(loc='outside upper center', ncols=3)
plt.show()
```

Note that for each tested value of the component separation, the analysis proceeds in three steps:
1. Create a `jc.PulseSpec` object, which describes the amplitude, location, and width of each Gaussian component in the profile, along with the corresponding modulation index and jitter parameter
2. Create a `jc.ProfileModel` object, which, in addition to the `PulseSpec`, contains information about the number of profiles to be generated, the number of pulses to average to form each profile, the phase resolution, signal-to-noise ratio, etc.
3. Call the `jc.analysis.run_analyses()` function with the `ProfileModel` object and a list of Analysis objects to run. This high-level interface takes care of generating a training dataset, training a model (e.g., by forming a template and principal components), generating a test dataset, and carrying out TOA inference.

The lower-level Estimator API allows for more control over the order of operations. Carrying out a single analysis using the Estimator API might look like this:

```python
import numpy as np
import matplotlib.pyplot as plt
import jitter_correction as jc

spec = jc.PulseSpec.new(
    amplitude = [0.8, 1.0, 0.4],
    loc = [-0.08, 0.00, 0.06],
    fwhm = [0.04, 0.06, 0.05],
    fj = [0.0, 0.0, 0.0],
    modindex = [1.0, 1.0, 1.0],
)
phase = np.linspace(-0.5, 0.5, 2048, endpoint=False)
true_template = spec.template(phase)

training_model = jc.ProfileModel(
    spec,
    n_profiles=512,
    npprof=1000,
    n_bins=2048,
    snr=1000,
    ampl_dist='lognorm',
    drift_bins=50,
)
training_data = training_model.generate_data()

result = jc.extract_pcs(training_data, n_pcs=8, use_trend=True)
pca_model, scores, dtoas = result

jc.plot_pcs(pca_model, 8)
plt.show()

validation_model = jc.ProfileModel(
    spec,
    n_profiles=512,
    npprof=1000,
    n_bins=2048,
    snr=1000,
    ampl_dist='lognorm',
    drift_bins=-40,
)
validation_data = validation_model.generate_data()

tm_estimator = jc.TemplateMatchingEstimator(pca_model.template)
tm_results = tm_estimator.estimate_toas(validation_data)
tm_dtoas = tm_results.toa - validation_model.shifts

pcm_estimator = jc.PCMatchingEstimator(pca_model)
pcm_results = pcm_estimator.estimate_toas(validation_data)
pcm_dtoas = pcm_results.toa - validation_model.shifts

profile_number = np.arange(validation_data.n_profiles)
fig, ax = plt.subplots()
ax.scatter(profile_number, tm_dtoas, marker='.')
ax.scatter(profile_number, pcm_dtoas, marker='.')
ax.set_xlabel("Profile number")
ax.set_ylabel(r"$\Delta$TOA (phase bins)")
plt.show()
```

As in the previous example, we begin by creating `PulseSpec` and `ProfileModel` objects, but in this case we handle each later step in the analysis directly. This involves:
1. Using the `jc.extract_pcs()` function to extract the principal components
2. Creating a separate `ProfileModel` (in this case with a different drift rate) for the validation data
3. Creating `TemplateMatchingEstimator` and `PCMatchingEstimator` objects and using their `estimate_toas()` methods to produce TOA estimates.

There are five different TOA estimation methods implemented in this package, including four different jitter correction methods, each with its own Estimator and Analysis classes. In short, the methods are:
1. Template matching (`TemplateMatchingEstimator`, `TemplateMatchingAnalysis`)
2. Skewness regression (`SkewnessRegressionEstimator`, `SkewnessRegressionAnalysis`)
3. Principal component regression (`PCRegressionEstimator`, `PCRegressionAnalysis`)
4. Generalized template matching (`PCMatchingEstimator`, `PCMatchingAnalysis`)
5. Generalized template matching with a prior (`PCBayesianEstimator`, `PCBayesianAnalysis`).

More detailed information can be found in the docstrings for the classes, functions, and methods mentioned above.
## Type checking
The code for this package has been fully type-checked using `mypy`. The type checker can be re-run using
```
pixi run type-check
```
