# CCLONE package

This package provides functions for analyis of single-cell variant call  data. We introduce CCLONE based on a weighted NMF.

The tools in this package are compatible with [scanpy](https://scanpy.readthedocs.io/).

### Instalation

To install the package from GitHub, please use:

     git clone https://github.com/HaghverdiLab/CCLONE
     cd CCLONE
     pip install -e .
     
### How to run

CCLONE takes as input an [Anndata](https://anndata.readthedocs.io) object with REF and ALT count matrices for variants saved in the Anndata.layers. 

Tutorials explaining the analysis workflow with CCLONE can be found [here](https://github.com/HaghverdiLab/CCLONE_notebooks), scripts to convert cellsnp-lite output to the needed Anndata object can also be found there.

### References

Valérie Marot-Lassauzaie, et al.

### Support

If you found a bug, would like to see a feature implemented, or have a question please open an [issue](https://github.com/HaghverdiLab/CCLONE/issues).
