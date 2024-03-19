# manuscript-binns_cortex_stn_comm

Code for the manuscript Binns *et al.* (Pre-print). DOI: TBC.

The repository has the following structure:
- [processing](processing) - location of scripts for (pre)processing of data (view [README](processing/README.txt))
- [analysis](analysis) - location of scripts for analysis of processed data (i.e. generating subject- and group-level results; view [README](analysis/README.txt))
- [plotting](plotting) - location of scripts for plotting results (view [README](plotting/README.txt)), with images stored in [figures](figures)
- [Project](project) - location of settings files and results (view [README](Project/README.txt))
- [coherence](coherence) - internal scripts for performing the analyses

\
The scripts have several external dependencies:
- for Python, these should be acquired using the [`environment.yml`](environment.yml) file with [conda](https://conda.io/projects/conda/en/latest/index.html), e.g.: 
`conda env create --file=environment.yml --solver=libmamba`
- for MATLAB (R2022b), these must be acquired manually:
  - [SPM12 r7771](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
  - [Lead-DBS v2.6](https://www.lead-dbs.org/)
  - [WJN Toolbox (tsbinns) v1.0.0](https://github.com/neuromodulation/wjn_toolbox_tsbinns/tree/1.0.0)

\
The scripts expect data to be in the [BIDS](https://bids.neuroimaging.io/) format.

The data used in the study can be made available conditionally to data sharing agreements in accordance with data privacy statements signed by the patients within the legal framework of the General Data Protection Regulation of the European Union.\
For this, you should contact the corresponding author: [julian.neumann@charite.de](mailto:julian.neumann@charite.de)
