import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mycolors = ["#797D62", "#9B9B7A", "#D9AE94", "#FFCB69", "#D08C60", "#997B66"]
BASE_PATH = "."

all_floors = glob.glob(f"{BASE_PATH}/input/indoor-location-navigation/metadata/*/*")
all_site_test = glob.glob(f"{BASE_PATH}/input/indoor-navigation-and-location-wifi-features/wifi_features/test/*")
floor_no = []
site_id_test = []

def show_values_on_bars(axs, h_v="v", space=0.4):
    '''Plots the value at the end of the a seaborn barplot.
    axs: the ax of the plot
    h_v: weather or not the barplot is vertical/ horizontal'''

    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, format(value, ','), ha="center")
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, format(value, ','), ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

for site in all_site_test:
    site_info = site.split("/")[5]
    site_id = site_info.split("_")[0]
    site_id_test.append(site_id)

# Extract only the floor number
for floor in all_floors:
    floor_site_id = floor.split("/")[4]
    if floor_site_id in site_id_test:
        no = floor.split("/")[5]
        floor_no.append(no)

floor_no = pd.DataFrame(floor_no, columns=["No"])
floor_no = floor_no["No"].value_counts().reset_index()
floor_no = floor_no.sort_values("No", ascending=False)

# ~~~~
# PLOT
# ~~~~
plt.figure(figsize=(16, 12))
ax = sns.barplot(data=floor_no, x="No", y="index", palette="Greens_r",
                 saturation=0.4)
show_values_on_bars(ax, h_v="h", space=0.4)
ax.set_title("Frequency of Floors", size=26, color=mycolors[0], weight='bold')
ax.set_xlabel("")
ax.set_ylabel("Floor No.", size=18, color=mycolors[0], weight='bold')
plt.xticks([])
plt.yticks(fontsize=11)
sns.despine(left=True, bottom=True);


