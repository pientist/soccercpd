import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from src.myconstants import *

pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
plt.rcParams['font.size'] = 15


class FormManager:
    def __init__(self, form_periods, role_records=None):
        self.form_periods = form_periods
        self.role_records = role_records

    @staticmethod
    def delaunay_dist(form1, form2):
        cost_mat = distance_matrix(form1['coords'], form2['coords'])
        _, perm = linear_sum_assignment(cost_mat)
        edge_mat1 = form1['edge_mat']
        edge_mat2 = form2['edge_mat'][perm][:, perm]
        return np.abs(edge_mat1 - edge_mat2).sum()

    def align_group(self, group, group_type=LABEL_FORMATION):
        form_periods = self.form_periods[self.form_periods[group_type] == group].reset_index()
    
        role_aligns = pd.DataFrame(np.vstack(form_periods[LABEL_COORDS].values), columns=[LABEL_X, LABEL_Y])
        coloring_model = AgglomerativeClustering(n_clusters=10).fit(role_aligns.values)
        role_aligns[LABEL_ACTIVITY_ID] = 0
        role_aligns[LABEL_FORM_PERIOD] = 0
        role_aligns[LABEL_BASE_ROLE] = np.repeat(np.arange(10)[np.newaxis, :] + 1, form_periods.shape[0], axis=0).flatten()
        role_aligns[LABEL_ALIGNED_ROLE] = coloring_model.labels_
        mean_coords = role_aligns.groupby(LABEL_ALIGNED_ROLE)[[LABEL_X, LABEL_Y]].mean().values
        
        for i in range(3):
            for i, coords in enumerate(form_periods[LABEL_COORDS]):
                assign_cost_mat = distance_matrix(mean_coords, coords)
                _, perm = linear_sum_assignment(assign_cost_mat)
                role_aligns.loc[perm+10*i, LABEL_ALIGNED_ROLE] = np.arange(10) + 1
                role_aligns.loc[perm+10*i, LABEL_ACTIVITY_ID] = form_periods.at[i, LABEL_ACTIVITY_ID]
                role_aligns.loc[perm+10*i, LABEL_FORM_PERIOD] = form_periods.at[i, LABEL_FORM_PERIOD]
            mean_coords = role_aligns.groupby(LABEL_ALIGNED_ROLE)[[LABEL_X, LABEL_Y]].mean()
        
        mean_coords['center_dist'] = np.linalg.norm(mean_coords, axis=1)

        mean_coords_df = mean_coords[(mean_coords['center_dist'] >= 1200) & (mean_coords['x'] < 0)]
        mean_coords_dm = mean_coords[(mean_coords['center_dist'] < 1200) & (mean_coords['x'] < 0)]
        mean_coords_am = mean_coords[(mean_coords['center_dist'] < 1200) & (mean_coords['x'] >= 0)]
        mean_coords_fw = mean_coords[(mean_coords['center_dist'] >= 1200) & (mean_coords['x'] >= 0)]

        roles_from = pd.concat([
            mean_coords_df.sort_values(LABEL_Y, ascending=False),
            mean_coords_dm.sort_values(LABEL_Y, ascending=False),
            mean_coords_am.sort_values(LABEL_Y, ascending=False),
            mean_coords_fw.sort_values(LABEL_Y, ascending=False)
        ]).index.tolist()
        role_dict = dict(zip(roles_from, np.arange(10) + 1))

        role_aligns[LABEL_ALIGNED_ROLE] = role_aligns[LABEL_ALIGNED_ROLE].replace(role_dict)
        return role_aligns[HEADER_ROLE_ALIGNS]

    def align(self, group_type=LABEL_FORMATION):
        role_aligns_list = []
        for f in np.sort(self.form_periods[group_type].unique()):
            role_aligns_list.append(self.align_group(f, group_type))
            print(f"Roles aligned for {group_type} '{f}'")

        role_aligns = pd.concat(role_aligns_list)
        self.role_records = pd.merge(pd.merge(
            self.role_records, self.form_periods[[LABEL_ACTIVITY_ID, LABEL_FORM_PERIOD, LABEL_FORMATION]]
        ), role_aligns).sort_values([LABEL_ACTIVITY_ID, LABEL_ROLE_PERIOD, LABEL_SQUAD_NUM], ignore_index=True)

    def visualize_group(self, group, group_type=LABEL_FORMATION, paint=True, annotate=True):
        if LABEL_ALIGNED_ROLE in self.role_records.columns:
            role_records = self.role_records[self.role_records[group_type] == group]
        else:
            role_records = self.align_group(group, group_type)
        colors = role_records[LABEL_ALIGNED_ROLE] if paint else 'gray'

        plt.figure()
        plt.scatter(role_records[LABEL_X], role_records[LABEL_Y], s=100, alpha=0.5, c=colors, cmap='tab10', zorder=0)

        if annotate:
            mean_coords = role_records.groupby(LABEL_ALIGNED_ROLE)[[LABEL_X, LABEL_Y]].mean()
            plt.scatter(mean_coords[LABEL_X], mean_coords[LABEL_Y], s=500, c='w', edgecolors='k', zorder=1)
            for r in mean_coords.index:
                plt.annotate(r, xy=mean_coords.loc[r], ha='center', va='center', fontsize=15, zorder=2)

        xlim = 3000
        ylim = 3000
        plt.xlim(-xlim, xlim)
        plt.ylim(-ylim, ylim)
        plt.vlines([-xlim, 0, xlim], ymin=-ylim, ymax=ylim, color='k', zorder=0)
        plt.hlines([-ylim, ylim], xmin=-xlim, xmax=xlim, color='k', zorder=0)
        plt.axis('off')

    def visualize(self, group_type=LABEL_FORMATION, paint=True, annotate=True, save=False):
        counts = self.form_periods[group_type].value_counts()
        for f in np.sort(self.form_periods[group_type].unique()):
            self.visualize_group(f, group_type, paint, annotate)
            if save:
                plt.savefig(f'img/clustering_{f}.pdf', bbox_inches='tight') 
            plt.title(f"{group_type} '{f}' - {counts[f]} periods.")

