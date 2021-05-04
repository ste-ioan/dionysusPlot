from os import listdir
import pandas as pd
import seaborn as sns
import umap
from bokeh.models import CustomJS, TextInput, RangeSlider, HoverTool
from bokeh.models import Legend, LegendItem, LinearAxis, CategoricalColorMapper, ColumnDataSource
from bokeh.layouts import column
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Spectral10
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

data_folder = 'C:\\Users\\sadek\\PycharmProjects\\wineScraper\\data\\'
files = [f for f in listdir(data_folder) if f.endswith('.csv')]

wines = []
for filename in files:
    wines.append(pd.read_csv(data_folder + '\\' + filename))

wines = pd.concat(wines)
wines.reset_index(inplace=True, drop=True)

# find the rare taste features (in less than 5 percent of the sample), drop them from the df
rare_features = wines.isnull().sum(axis=0)[wines.isnull().sum(axis=0) > (wines.shape[0] * 0.95)].index.to_list()

# remove wrong features
undesirables = ['region', 'cheese', 'game', 'meat']

# tiny issue here 'in' will return true also w partial string, would like to do only by whole word
for tastefeat in wines.columns.to_list():
    for word in undesirables:
        if word in tastefeat.lower():
            rare_features.append(tastefeat)

wines.drop(rare_features, axis=1, inplace=True)

# UMAP(a=None, angular_rp_forest=False, b=None,
#     force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
#     local_connectivity=1.0, low_memory=False, metric='euclidean',
#     metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
#     n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
#     output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
#     set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
#     target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
#    transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)

# find index for taste features
res = [i for i, val in enumerate(wines.columns == 'Rating') if val]

# establish a cutoff to exclude wines with too few..
# highest number of comments across taste feature in each bottle?
wines[wines.columns[res[0] + 1:].to_list()].max(axis=1)
# the mean is skewed but the median might be an acceptable cutoff, if a wine has less than
# that number of comments in its highest scoring feature, then we'll filter it out
# filtro = wines[wines.columns[res[0]+1:].to_list()].max(axis=1).median()

# no e' meglio per ora un filtro arbitrario (la feature piu' votata deve avere almeno x voti)
filtro = 30
wines.drop(wines[wines[wines.columns[res[0] + 1:].to_list()].max(axis=1) < filtro].index, inplace=True)

# stash the taste features, normalise and reinput them into the matrix, fill na's with 0s
taste_features_columns = wines.columns[res[0] + 1:].to_list()
normalised_tastes = (wines[taste_features_columns] - wines[taste_features_columns].mean()) / wines[
    taste_features_columns].std()
wines[taste_features_columns] = normalised_tastes.fillna(0)
# give avg alcohol content to wines that are missing this data
wines['Alcohol'].fillna(wines['Alcohol'].mean().round(1), inplace=True)

# remove any rows that still have nas (no slider data for those wines probably)
wines.dropna(inplace=True)
wines.reset_index(inplace=True, drop=True)

hover_data = pd.DataFrame({'name': wines['wine'].str.lower(),
                           'price': wines['Price'],
                           'grapes': wines['Grapes'].str.lower()})



wines.drop(['Price', 'Rating'], axis=1, inplace=True)

# print out some info
print("There are {} wines and {} taste features in the dataset".format(wines.shape[0], wines.shape[1]))

# another (faster) way of doing things
mapper = umap.UMAP().fit(wines[wines.columns[3:].to_list()])

# this guy automatically makes the legend with labels, interactive plot doesn't and haven't found
# way to patch it do so, since it would need every data group to be plotted subsequently
# umap.plot.points(mapper, labels=wines['Variety'])
import umap.plot
p = umap.plot.interactive(mapper, labels=wines['Variety'],
                          hover_data=hover_data, point_size=6.5, interactive_text_search=True)

data = pd.DataFrame(mapper.embedding_, columns=("x", "y"))
data['label'] = wines['Variety']
unique_labels = np.unique(data['label'])
num_labels = unique_labels.shape[0]
color_key = _to_hex(plt.get_cmap('Spectral')(np.linspace(0, 1, num_labels)))
new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
data["color"] = pd.Series(wines['Variety']).map(new_color_key)
data["alpha"] = 1
datasource = ColumnDataSource(data)

tooltip_dict = {}
for col_name in hover_data:
    data[col_name] = hover_data[col_name]
    tooltip_dict[col_name] = "@{" + col_name + "}"
tooltips = list(tooltip_dict.items())

output_file('red_wines_struct.html')

plot_figure = figure(
    title='UMAP projection of the Red Wines',
    plot_width=600,
    plot_height=600,
    tooltips=tooltips,
    background_fill_color='white',
)

plot_figure.circle(
    'x',
    'y',
    source=datasource,
    color='color',
    alpha='alpha',
    size=6.5,
)

print('hi')
'''
show(column(searchbar, plot_figure, slider))


these inputs won't work he doesnt recognize the plot to show..
interactive_text_search=True,
                          interactive_text_search_columns=wines['wine']


umap.

# configure visual properties on a plot's title attribute
p.title.text = "Red wines space"
p.title.align = "left"
p.title.text_color = "red"
p.title.text_font_size = "25px"
p.legend.location = "top_left"
# p.toolbar_location = "above"
# p.title.background_fill_color = "#aaaaee"

text_input = TextInput(value="", title="Search wine:")
callback = CustomJS(
    args=dict(
        source=hover_data.to_json(),
        matching_alpha=0.95,
        non_matching_alpha=1 - 0.95,
        search_columns='name',
    ),
    code="")


text_input.js_on_change("value", callback)
'''

'''
# take only flavors, without sliders changes the clusters by a lot
mapper = umap.UMAP().fit(wines[wines.columns[11:].to_list()])

import umap.plot
# umap.plot.points(mapper, labels=wines['Variety'])

umap.plot.output_file('red_wines_not_struct.html.html')

p = umap.plot.interactive(mapper, labels=wines['Variety'],
                          hover_data=hover_data, point_size=5)
umap.plot.show(p)


digits_df = pd.DataFrame(embedding, columns=('x', 'y'))
digits_df['digit'] = [str(x) for x in digits.target]
digits_df['image'] = list(map(embeddable_image, digits.images))



plot_figure = figure(
    title='UMAP projection of the Digits dataset',
    plot_width=600,
    plot_height=600,
    tools=('pan, wheel_zoom, reset')
)

plot_figure.add_tools(HoverTool(tooltips="""
<div>
    <div>
        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>Digit:</span>
        <span style='font-size: 18px'>@digit</span>
    </div>
</div>
"""))

plot_figure.circle(
    'x',
    'y',
    source=datasource,
    color=dict(field='digit', transform=color_mapping),
    line_alpha=0.6,
    fill_alpha=0.6,
    size=4
)
show(plot_figure)

'''