<div align="center">
	<h1>FootballCPD: Formation and Role Change-Point Detection in Football Matches from Spatiotemporal Tracking Data</h1>
	<br>
</div>

**FootballCPD** is a change-point detection framework that distinguishes tactically intended formation and role changes from temporary changes in football matches using spatiotemporal tracking data.<br>
It first assigns roles to players frame-by-frame based on the role representation (Bialkowski et al., 2014) and performs two-step nonparametric change-point detections (Song and Chen, 2020): (1) formation change-point detection (FormCPD) based on the sequence of role-adjacency matrices and (2) role change-point detection (RoleCPD) based on the sequence of role permutations.<br>

Here is an example of applying FootballCPD to a match. It shows that the match is split into four formation periods (where the team formation is consistent) and five role periods (where the player-role assignment remains constant).<br>

![timeline](img/timeline_formation.png)<br>

## References
- Bialkowski, A., Lucey, P., Carr, P., Yue, Y., Sridharan,S., and Matthews, I. (2014). Large-scale analysis of soccer matches using spatiotemporal tracking data. In IEEE International Conference on Data Mining.
- Song, H. and Chen, H. (2020). Asymptotic distribution-free change-point detection for data with repeated observations.
