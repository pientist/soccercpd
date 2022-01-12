<div align="center">
	<h1>
		FootballCPD
	</h1>
</div>

## Original Paper
This repository includes the source code for the following paper and tracking data from a sample match (`245.ugp`). Please cite when using our code or the sample match data.
- **FootballCPD: Formation and role change-point detection in football matches from spatiotemporal tracking data**, Hyunsung Kim, Bit Kim, Dongwook Chung, Jinsung Yoon, and Sang-Ki Ko, 2021.

## Introduction
**FootballCPD** is a change-point detection framework that distinguishes tactically intended formation and role changes from temporary changes in football matches using spatiotemporal tracking data.<br>
It first assigns roles to players frame-by-frame based on the role representation (Bialkowski et al., 2014) and performs two-step nonparametric change-point detections (Song and Chen, 2020): (1) formation change-point detection (FormCPD) based on the sequence of role-adjacency matrices and (2) role change-point detection (RoleCPD) based on the sequence of role permutations.<br>

Here is an example of applying FootballCPD to a match. It shows that the match is split into four formation periods (where the team formation is consistent) and five role periods (where the player-role assignment remains constant).<br>

![timeline](img/timeline_formation.png)<br>

We cannot share the entire dataset due to the security issue, but every process except for the formation clustering can be reproduced using the sample match data (`245.ugp`) since applying our method to a match does not require data from other matches. For the formation clustering step in Section 4.1.3 of the paper, we offer `data/form_periods.pkl` that contains the mean role locations and mean role-adjacency matrices of all the detected formations in our dataset.<br>

## Getting Started
We have implemented the role representation algorithm using Python on our own, while adopted the R package `gSeg` for discrete g-segmentation. Therefore, **both Python and R need to be installed for executing the code.** The version we have used in this study are as follows:

- Python 3.8
- R 3.6.0

To perform the whole process at once, we utilize the Python package `rpy2` to run the R script with  `gSeg` inside our Python implementation. We found that **the M1 chip by Apple does not support binaries for `rpy2`'s API mode in Python, raising a memory error.** (So please use another processor such as the Intel chip.)

After installing the necessary languages, you need to install the packages listed in `requirements.txt`. Make sure that you are in the correct working directory that contains our `requirements.txt`.
```
pip install -r requirements.txt
```

Subsequently, please download the sample match data (named `245.ugp`) from the following Google Drive link and move it into the directory `data/ugp`.
- Link for the data: https://bit.ly/3E9W3QV

Finally, you can run the algorithm for the sample match data simply by executing `src/main.py`.
```
python -m src.main
```

For further analyses, open `tutorial.ipynb` and run the cells in order from the above. (Section 4 except for 4.1.3)

You can also reproduce the results described in the paper by executing the following notebooks:

- `src/0_formaiont_clustering.ipynb` (Section 4.1.3)
- `src/1_model_evaluation.ipynb` (Section 5.1)
- `src/2_switching_pattern_discovery.ipynb` (Section 5.2)
- `src/supplement_case_study.ipynb` (Section 2 in the supplementary paper)<br>

Lastly, we visualize our results as animations using Tableau 2020.4. The full-version videos are available in https://bit.ly/3GaaAOc (animation for a single team) and https://bit.ly/3m2XUR5 (animation for both teams competing in a match) with the description in Section 3 in the supplementary paper.<br>

## References
- Bialkowski, A., Lucey, P., Carr, P., Yue, Y., Sridharan, S., and Matthews, I. (2014). Large-scale analysis of soccer matches using spatiotemporal tracking data. In IEEE International Conference on Data Mining.
- Song, H. and Chen, H. (2020). Asymptotic distribution-free change-point detection for data with repeated observations.
