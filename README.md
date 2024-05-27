# manuscript-binns_cortex_stn_comm

Code for the manuscript Binns *et al.* (Pre-print). DOI: [10.1101/2024.04.14.586969](https://doi.org/10.1101/2024.04.14.586969).

The repository has the following structure:
- [processing](processing) - location of scripts for (pre)processing of data (view [README](processing/README.txt))
- [analysis](analysis) - location of scripts for analysis of processed data (i.e. generating subject- and group-level results; view [README](analysis/README.txt))
- [plotting](plotting) - location of scripts for plotting results (view [README](plotting/README.txt)), with images stored in [figures](figures)
- [Project](project) - location of settings files and results (view [README](Project/README.txt))
- [coherence](coherence) - internal scripts for performing the analyses
- [DEMO](DEMO) - location of simulated data and example scripts for performing the analyses (view [README](DEMO/README.txt))

\
The scripts have several external dependencies (the exact versions should be followed to ensure reproducibility):
- for Python, these should be acquired using the [`environment.yml`](environment.yml) file with [conda](https://conda.io/projects/conda/en/latest/index.html). This will not take more than a few minutes if the `libmamba` solver is used, e.g.: 
  ```
  conda env create --file=environment.yml --solver=libmamba
  ```
- for MATLAB (R2022b), these must be acquired manually (this will not take more than a few minutes):
  - [SPM12 r7771](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
  - [Lead-DBS v2.6](https://www.lead-dbs.org/)
  - [WJN Toolbox (tsbinns) v1.0.0](https://github.com/neuromodulation/wjn_toolbox_tsbinns/tree/1.0.0)

Testing for all code has been performed with Windows 11 and the package versions specified above.

\
The scripts expect data to be in the [BIDS](https://bids.neuroimaging.io/) format.

\
The data used in the study can be made available conditionally to data sharing agreements in accordance with data privacy statements signed by the patients within the legal framework of the General Data Protection Regulation of the European Union. For this, you should contact the corresponding author ([julian.neumann@charite.de](mailto:julian.neumann@charite.de)) or the Open Data Officer ([opendata-neuromodulation@charite.de](mailto:opendata-neuromodulation@charite.de)).
