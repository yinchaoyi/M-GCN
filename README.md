# M-GCN
Molecular Subtyping of Cancer based on Robust Graph Neural Network and Multi-omics Data Integration
Code
---
`experiment.py` is main program. To run the code,  `python experiment.py`  
Data
---
The data is available from TCGA (https://portal.gdc.cancer.gov/). The input contains gene expression,  single nucleotide variants (SNV) and copy number variation (CNV) data. Label is the molecular subtyping of the sample. We apply our model on breast cancer (BRCA) and stomach adenocarcinoma (STAD). BRCA includes molecular subtypes of ER+, HER2+ and TNBC, and STAD includes molecular subtypes of CIN, EBV, MSI and GS, respectively.
