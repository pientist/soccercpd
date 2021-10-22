<div align="center">
	<h1>FootballCPD: Formation and Role Change-Point Detection in Football Matches from Spatiotemporal Tracking Data</h1>
	<br>
</div>

## Introduction
**FootballCPD** is a change-point detection framework that distinguishes tactically intended formation and role changes from temporary changes in football matches using spatiotemporal tracking data.<br>
It first assigns roles to players frame-by-frame based on the role representation (Bialkowski et al., 2014) and performs two-step nonparametric change-point detections (Song and Chen, 2020): (1) formation change-point detection (FormCPD) based on the sequence of role-adjacency matrices and (2) role change-point detection (RoleCPD) based on the sequence of role permutations.<br>

Here is an example of applying FootballCPD to a match. It shows that the match is split into four formation periods (where the team formation is consistent) and five role periods (where the player-role assignment remains constant).<br>

![timeline](img/timeline_formation.png)<br>

This repository includes the source code for FootballCPD, and tracking data from a sample match (`data/ugp/245.ugp`).
We cannot share the entire dataset due to the security issue, but every process except for the formation clustering can be reproduced using this sample data since applying our method to a match does not require data from other matches. For the formation clustering step in Section 4.1.3 of the paper, we offer `data/form_periods.pkl` that contains the mean role locations and mean role-adjacency matrices of all the detected formations in our dataset.<br>

## Getting Started
After pulling this repository, you first need to install necessary packages listed in `requirements.txt`.
```
pip install -r requirements.txt
```
Make sure that you are in the correct working directory that contains our `requirements.txt`.

Subsequently, you can run the algorithm for the sample match data (`data/ugp/245.ugp`) simply by executing the `main.py` module.
```
python -m main
```
For further analyses, open `tutorial.ipynb` and run the cells in order from the above. (Section 4 except for 4.1.3)

You can also reproduce the results described in the paper by executing the following notebooks:
- `src/experiment0_formaiont_clustering.ipynb` (Section 4.1.3)
- `src/experiment1_model_evaluation.ipynb` (Section 5.1)
- `src/experiment2_switching pattern discovery.ipynb` (Section 5.2)
- `src/supplement_case_study.ipynb` (Section 2 in the supplementary paper)<br>

## Implementation Details
We have implemented the role representation algorithm using **Python 3.8** on our own, while adopted the R package `gSeg` for discrete g-segmentation. To perform the whole process at once, we utilize the Python package `rpy2` to run the R script with  `gSeg` inside our Python implementation. The process was executed on an Intel Core i7-8550U CPU with 16.0GB of memory. **Note that the M1 chip by Apple does not support binaries for `rpy2`'s API mode in Python, raising a memory error.**<br>

## References
- Bialkowski, A., Lucey, P., Carr, P., Yue, Y., Sridharan, S., and Matthews, I. (2014). Large-scale analysis of soccer matches using spatiotemporal tracking data. In IEEE International Conference on Data Mining.
- Song, H. and Chen, H. (2020). Asymptotic distribution-free change-point detection for data with repeated observations.
