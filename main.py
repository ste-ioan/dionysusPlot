from os import listdir
import sys
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

data_folder = ""

try:
    data_folder = 'C:\\Users\\sadek\\PycharmProjects\\wineScraper\\data\\'

except:
    print("Could not find files directory...")

if data_folder != "C:\\Users\\sadek\\PycharmProjects\\wineScraper\\data\\":
    data_folder = sys.argv[1]

files = [f for f in listdir(data_folder) if f.endswith('.csv')]

wines = []
for filename in files:
    wines.append(pd.read_csv(data_folder + '\\' + filename))

wines = pd.concat(wines)
wines.reset_index(inplace=True, drop=True)

# sum some redundant features
wines['Anise'] = (wines['Anise'].fillna(0) + wines['Star anise'].fillna(0)).replace({'0':np.nan, 0:np.nan})
wines['Tropical'] = (wines['Tropical'].fillna(0) + wines['Mango'].fillna(0) + wines['Pineapple'].fillna(0) + wines['Green papaya'].fillna(0) +
                     wines['Passion fruit'].fillna(0) + wines['Guava'].fillna(0) + wines['Green mango'].fillna(0)).replace({'0':np.nan, 0:np.nan})
wines['Orange'] = (wines['Orange'].fillna(0) + wines['Blood orange'].fillna(0)).replace({'0':np.nan, 0:np.nan})
wines['Smoke'] = (wines['Smoke'].fillna(0) + wines['Campfire'].fillna(0)).replace({'0':np.nan, 0:np.nan})
wines['Orange zest'] = (wines['Orange zest'].fillna(0) + wines['Orange peel'].fillna(0) + wines['Orange rind'].fillna(0)).replace({'0':np.nan, 0:np.nan})
wines['Dark fruit'] = (wines['Dark fruit'].fillna(0) + wines['Black fruit'].fillna(0)).replace({'0':np.nan, 0:np.nan})
wines['Dried flowers'] =  (wines['Dried flowers'].fillna(0) + wines['Dried rose'].fillna(0) + wines['Potpourri'].fillna(0)).replace({'0':np.nan, 0:np.nan})


wines.drop(['Star anise', 'Mango', 'Pineapple', 'Passion fruit', 'Orange peel', 'Guava', 'Green mango',
            'Green papaya', 'Orange rind', 'Blood orange', 'Campfire', 'Black fruit', 'Dried rose', 'Potpourri'], axis=1, inplace=True)

# find the rare taste features (in less than 10 percent of the sample), drop them from the df
rare_features = wines.isnull().sum(axis=0)[wines.isnull().sum(axis=0) > (wines.shape[0] * 0.90)].index.to_list()

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

# let's normalise also the slider data (sweet, dry, etc)
wines[wines.columns[3:11]] = (wines[wines.columns[3:11]] - wines[wines.columns[3:11]].mean()) / wines[wines.columns[3:11]].std()

# give avg alcohol content to wines that are missing this data, by variety
for variety in wines['Variety'].unique():
    wines[wines['Variety'] == variety]['Alcohol'].fillna(wines[wines['Variety'] == variety]['Alcohol'].mean(), inplace=True)
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

data = pd.DataFrame(mapper.embedding_, columns=("x", "y"))
data['label'] = wines['Variety']
unique_labels = np.unique(data['label'])
num_labels = unique_labels.shape[0]
color_key = _to_hex(plt.get_cmap('Spectral')(np.linspace(0, 1, num_labels)))
new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
data["color"] = pd.Series(wines['Variety']).map(new_color_key)
data["alpha"] = 1
data["wine_name"] = wines["wine"].str.lower()

tooltip_dict = {}
for col_name in hover_data:
    data[col_name] = hover_data[col_name]
    tooltip_dict[col_name] = "@{" + col_name + "}"
tooltips = list(tooltip_dict.items())

data_source = ColumnDataSource(data)

print("Creating struct file...")
output_file('red_wines_struct_new.html')

plot_figure = figure(
    title='UMAP projection of the Red Wines alpha',
    plot_width=600,
    plot_height=600,
    tooltips=tooltips,
    background_fill_color='white',
)

plot_figure.circle(
    'x',
    'y',
    source=data_source,
    color='color',
    alpha='alpha',
    size=6.5,
)

plot_figure.grid.visible = False
plot_figure.axis.visible = False

###########

text_input = TextInput(value="", title="Search for wine:")
callback = CustomJS(
args=dict(
    source=data_source,
    matching_alpha=1,
    non_matching_alpha=1 - 0.95,
    search_columns=["wine_name"],
),
code="""
var data = source.data;
var text_search = cb_obj.value;

var search_columns_dict = {}
for (var col in search_columns){
    search_columns_dict[col] = search_columns[col]
}

// Loop over columns and values
// If there is no match for any column for a given row, change the alpha value
var string_match = false;
for (var i = 0; i < data.x.length; i++) {
    string_match = false
    for (var j in search_columns_dict) {
        if (String(data[search_columns_dict[j]][i]).includes(text_search) ) {
            string_match = true
        }
    }
    if (string_match){
        data['alpha'][i] = matching_alpha
    }else{
        data['alpha'][i] = non_matching_alpha
    }
}
source.change.emit();
""",
)

text_input.js_on_change("value", callback)

plot_figure = column(text_input, plot_figure)






###########

show(plot_figure)

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
'''
