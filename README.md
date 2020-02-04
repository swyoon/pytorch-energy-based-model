# pytorch-energy-based-model
This repository provides simple illustrative working examples for energy-based models (EBM) in PyTorch.

The aim of the repository is to provide educational resources, to validate each step with toy examples, and to build a platform for future experiment.

## Quickstart

The main requirements are `python>=3.6` and `torch>=1.2`.

```
pip install -r requirements.txt
```

**Validate Langevin dynamics sampling**
```
python run_langevin.py 8gaussians
```

**Training an energy-based model**
```
python run_ebm.py 8gaussians
```


## Expected Results


## Directories

* `run_langevin.py` : Run Langevin dynamics sampling of a toy distribution. Produces images of samples.
* `run_ebm.py` : Train an EBM for a samples from a toy distribution.
* `langevin.py` : Codes related to Langevin dynamics
* `model.py` : Codes related to neural networks
* `data.py` : Codes related to generating toy distributions


## Further reading

* IGEMB
* LeCun
* secretely
