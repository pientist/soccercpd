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
plt.rcParams['font.family'] = 'Arial'


class FormManager:
    def __init__(self, form_periods, role_records=None):
        self.form_periods = form_periods
        self.role_records = role_records

    @staticmethod
    def align_group(form_periods):
        role_aligns = pd.DataFrame(np.vstack(form_periods[LABEL_COORDS].values), columns=[LABEL_X, LABEL_Y])
        coloring_model = AgglomerativeClustering(n_clusters=10).fit(role_aligns.values)

        role_aligns[LABEL_ACTIVITY_ID] = 0
        role_aligns[LABEL_FORM_PERIOD] = 0

        base_roles_repeated = np.repeat(np.arange(10)[np.newaxis, :] + 1, form_periods.shape[0], axis=0)
        role_aligns[LABEL_BASE_ROLE] = base_roles_repeated.flatten()
        role_aligns[LABEL_ALIGNED_ROLE] = coloring_model.labels_

        mean_coords = role_aligns.groupby(LABEL_ALIGNED_ROLE)[[LABEL_X, LABEL_Y]].mean().values
        
        for _ in range(3):
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
        for group in np.sort(self.form_periods[group_type].unique()):
            form_periods = self.form_periods[self.form_periods[group_type] == group].reset_index()
            role_aligns_list.append(FormManager.align_group(form_periods))
            print(f"Roles aligned for {group_type} '{group}'")

        role_aligns = pd.concat(role_aligns_list)[HEADER_ROLE_ALIGNS[:-2]]
        self.role_records = pd.merge(
            self.role_records[HEADER_ROLE_RECORDS],
            self.form_periods[[LABEL_ACTIVITY_ID, LABEL_FORM_PERIOD, LABEL_FORMATION]]
        )
        self.role_records = pd.merge(
            self.role_records, role_aligns
        ).sort_values([LABEL_ACTIVITY_ID, LABEL_ROLE_PERIOD, LABEL_SQUAD_NUM], ignore_index=True)

    @staticmethod
    def visualize_single_graph(coords, edge_mat, labels=None):
        plt.figure(figsize=(7, 5))
        plt.scatter(coords[:, 0], coords[:, 1], c=np.arange(10)+1,
                    s=1000, vmin=0.5, vmax=10.5, cmap='tab10', zorder=1)

        if labels is None:
            labels = np.arange(11)
            fontsize = 20
        else:
            fontsize = 15

        for i in np.arange(10):
            plt.annotate(labels[i+1], xy=coords[i], ha='center', va='center',
                         c='w', fontsize=fontsize, fontweight='bold', zorder=2)
            for j in np.arange(10):
                plt.plot(coords[[i, j], 0], coords[[i, j], 1],
                         linewidth=edge_mat[i, j] ** 2 * 10, c='k', zorder=0)

        xlim = 3000
        ylim = 2400
        # plt.xlim(-xlim - 500, xlim + 500)
        # plt.ylim(-ylim - 500, ylim + 500)
        plt.xlim(-xlim, xlim)
        plt.ylim(-ylim, ylim)
        plt.vlines([-xlim, 0, xlim], ymin=-ylim, ymax=ylim, color='k', zorder=0)
        plt.hlines([-ylim, ylim], xmin=-xlim, xmax=xlim, color='k', zorder=0)
        plt.axis('off')

    def visualize_group(self, group, group_type=LABEL_FORMATION, paint=True, annotate=True):
        if self.role_records is not None and LABEL_ALIGNED_ROLE in self.role_records.columns:
            role_records = self.role_records[self.role_records[group_type] == group]
        else:
            form_periods = self.form_periods[self.form_periods[group_type] == group]
            role_records = FormManager.align_group(form_periods.reset_index(drop=True))
        colors = role_records[LABEL_ALIGNED_ROLE] if paint else 'gray'

        plt.figure(figsize=(7, 5))
        plt.scatter(role_records[LABEL_X], role_records[LABEL_Y], s=150, alpha=0.5, c=colors, cmap='tab10', zorder=0)

        if annotate:
            mean_coords = role_records.groupby(LABEL_ALIGNED_ROLE)[[LABEL_X, LABEL_Y]].mean()
            plt.scatter(mean_coords[LABEL_X], mean_coords[LABEL_Y], s=1000, c='w', edgecolors='k', zorder=1)
            for r in mean_coords.index:
                plt.annotate(r, xy=mean_coords.loc[r], ha='center', va='center', fontsize=25, zorder=2)

        xlim = 3000
        ylim = 3000
        plt.xlim(-xlim, xlim)
        plt.ylim(-ylim, ylim)
        plt.vlines([-xlim, 0, xlim], ymin=-ylim, ymax=ylim, color='k', zorder=0)
        plt.hlines([-ylim, ylim], xmin=-xlim, xmax=xlim, color='k', zorder=0)
        plt.axis('off')

    def visualize(
        self, group_type=LABEL_FORMATION, ignore_outliers=False,
        paint=True, annotate=True, save=False
    ):
        counts = self.form_periods[group_type].value_counts()
        for group in np.sort(self.form_periods[group_type].unique()):
            if ignore_outliers and (group == -1 or group == 'others'):
                continue
            self.visualize_group(group, group_type, paint, annotate)
            if save:
                plt.savefig(f'img/{group_type}_{group}.png', bbox_inches='tight')
            title = f"{group_type[0].upper() + group_type[1:]} {group} -  {counts[group]} periods"
            plt.title(title)
