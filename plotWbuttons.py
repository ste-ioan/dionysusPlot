from os import listdir
import sys
import pandas as pd
import seaborn as sns
import umap
from bokeh.models import CustomJS, TextInput, RangeSlider, HoverTool
from bokeh.models import Legend, LegendItem, LinearAxis, CategoricalColorMapper, ColumnDataSource, CheckboxButtonGroup
from bokeh.layouts import column, row
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Spectral10
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import product


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
data_folder = 'C:\\Users\\sadek\\PycharmProjects\\wineScraper\\data\\merged\\Reds\\'

try:
    listdir(data_folder)

except:
    print("Could not find files directory...")
    data_folder = sys.argv[1]

files = [f for f in listdir(data_folder) if f.endswith('.csv')]

wines = []
for filename in files:
    wines.append(pd.read_csv(data_folder + '\\' + filename))

wines = pd.concat(wines)
wines.reset_index(inplace=True, drop=True)

# sum some redundant features
wines['Anise'] = (wines.loc[:, 'Anise'].fillna(0) + wines.loc[:, 'Star anise'].fillna(0)).replace(
    {'0': np.nan, 0: np.nan})
wines['Tropical'] = (
            wines.loc[:, 'Tropical'].fillna(0) + wines.loc[:, 'Mango'].fillna(0) + wines.loc[:, 'Pineapple'].fillna(
        0) + wines.loc[:, 'Green papaya'].fillna(0) +
            wines.loc[:, 'Passion fruit'].fillna(0) + wines.loc[:, 'Guava'].fillna(0) + wines.loc[:,
                                                                                        'Green mango'].fillna(
        0)).replace({'0': np.nan, 0: np.nan})
wines['Orange'] = (wines.loc[:, 'Orange'].fillna(0) + wines.loc[:, 'Blood orange'].fillna(0)).replace(
    {'0': np.nan, 0: np.nan})
wines['Smoke'] = (wines.loc[:, 'Smoke'].fillna(0) + wines.loc[:, 'Campfire'].fillna(0)).replace(
    {'0': np.nan, 0: np.nan})
wines['Orange zest'] = (wines.loc[:, 'Orange zest'].fillna(0) + wines.loc[:, 'Orange peel'].fillna(0) + wines.loc[:,
                                                                                                        'Orange rind'].fillna(
    0)).replace({'0': np.nan, 0: np.nan})
wines['Dark fruit'] = (wines.loc[:, 'Dark fruit'].fillna(0) + wines.loc[:, 'Black fruit'].fillna(0)).replace(
    {'0': np.nan, 0: np.nan})
wines['Dried flowers'] = (wines.loc[:, 'Dried flowers'].fillna(0) + wines.loc[:, 'Dried rose'].fillna(0) + wines.loc[:,
                                                                                                           'Potpourri'].fillna(
    0)).replace({'0': np.nan, 0: np.nan})

wines.drop(['Star anise', 'Mango', 'Pineapple', 'Passion fruit', 'Orange peel', 'Guava', 'Green mango',
            'Green papaya', 'Orange rind', 'Blood orange', 'Campfire', 'Black fruit', 'Dried rose', 'Potpourri'],
           axis=1, inplace=True)

# find the rare taste features (in less than 20 percent of the sample), drop them from the df
rare_features = wines.isnull().sum(axis=0)[wines.isnull().sum(axis=0) > (wines.shape[0] * 0.80)].index.to_list()

# remove wrong features
undesirables = ['region', 'cheese', 'game', 'meat']

# tiny issue here 'in' will return true also w partial string, would like to do only by whole word
for tastefeat in wines.columns.to_list():
    for word in undesirables:
        if word in tastefeat.lower():
            rare_features.append(tastefeat)

wines.drop(rare_features, axis=1, inplace=True)

# homogenise grape names
for grapes in wines['Grapes']:
    if type(grapes) == str:
        if len(grapes.split(',')) > 1:
            new_grapes = list()
            for grape in grapes.split(','):
                new_grapes.append(grape.strip())

            new_grapes.sort()
            wines.loc[wines['Grapes'] == grapes, 'Grapes'] = ', '.join(new_grapes)

# find index for taste features
res = [i for i, val in enumerate(wines.columns == 'Rating') if val]

# establish a cutoff to exclude wines with too few votes on the tastes
filtro = 30
wines.drop(wines[wines[wines.columns[res[0] + 1:].to_list()].fillna(0).max(axis=1) < filtro].index, inplace=True)

print("Votes from {} people were employed to estimate the taste features".format(wines.loc[:,wines.columns[res[0]+1:]].max(axis=1).sum()))

# stash the taste features, normalise and reinput them into the matrix, fill na's with 0s
taste_features_columns = wines.columns[res[0] + 1:].to_list()
normalised_tastes = wines[taste_features_columns].subtract(wines[taste_features_columns].mean(axis=1), axis=0).divide(
    wines[taste_features_columns].std(axis=1), axis=0)
wines[taste_features_columns] = normalised_tastes.fillna(0)

# find indices for slider taste features
first = [i for i, val in enumerate(wines.columns == 'Bold') if val]
last = [i for i, val in enumerate(wines.columns == 'Soft') if val]

# let's normalise also the slider data (sweet, dry, etc)
wines[wines.columns[first[0]:last[0] + 1]] = wines[wines.columns[first[0]:last[0] + 1]].subtract(
    wines[wines.columns[first[0]:last[0] + 1]].mean()).divide(wines[wines.columns[first[0]:last[0] + 1]].std())

# give avg alcohol content to wines that are missing this data, by specialty (this breaks down very separate clusters!)
# alternative is to give avg alcohol content, that brings it more together
for specialty in wines['Specialty'].unique():
    copy = wines.loc[wines.loc[:, 'Specialty'] == specialty, 'Alcohol']
    media = wines[wines['Specialty'] == specialty]['Alcohol'].mean()
    wines.loc[wines.loc[:, 'Specialty'] == specialty, 'Alcohol'] = copy.fillna(media)

# here normalise alcohol content
wines.loc[:, 'Alcohol'] = wines.loc[:, 'Alcohol'].subtract(wines.loc[:, 'Alcohol'].mean()).divide(
    wines.loc[:, 'Alcohol'].std())

wines['Price'].fillna(0, inplace=True)

wines.drop('Rating', axis=1, inplace=True)
# remove any rows that still have nas (no slider data for those wines probably), after fixing price
wines.dropna(inplace=True)
wines.reset_index(inplace=True, drop=True)

hover_data = pd.DataFrame({'specialty': wines['Specialty'].str.lower(),
                           'grapes': wines['Grapes'].str.lower(),
                           })

wines.drop('Price', axis=1, inplace=True)

# print out some info
print("There are {} wines and {} taste features in the dataset".format(wines.shape[0], wines.shape[1]))

# mapper parameters
fit = umap.UMAP(
    n_neighbors=10,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    transform_seed=42,
)

# fit the mapper
taste_data = wines[wines.columns[first[0]:].to_list()]
embedding = fit.fit_transform(taste_data)
# create data df for plot, compatible with widgets
data = pd.DataFrame(embedding, columns=("x", "y"))
# assign Variety as label and create a color code for each Variety
data['label'] = wines['Variety']
unique_labels = np.unique(data['label'])
num_labels = unique_labels.shape[0]
color_key = _to_hex(plt.get_cmap('Spectral')(np.linspace(0, 1, num_labels)))
new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
# include color as a column in the data df, along with some other info that will be used for the plot&widgets
data["color"] = pd.Series(data['label']).map(new_color_key)
data["alpha"] = 1
data["wine_name"] = wines["wine"].str.lower()
# data["price"] = hover_data["price"]

tooltip_dict = {}
for col_name in hover_data:
    data[col_name] = hover_data[col_name]
    tooltip_dict[col_name] = "@{" + col_name + "}"
tooltips = list(tooltip_dict.items())

buttons_data_source = ColumnDataSource(wines[wines.columns[3:]])
reference_data_by_fam = ColumnDataSource(wines.groupby('Specialty').mean())

print("Creating struct file...")
output_file('button_attempt.html')

##### CREATE PLOT
plot_figure = figure(
    title='Dionysus tastespace of {} red wines - alpha'.format(wines.shape[0]),
    plot_width=700,
    plot_height=700,
    tooltips=tooltips,
    background_fill_color='white',
)

for variety in unique_labels:
    data_source = ColumnDataSource(data.loc[data['label'] == variety, ])

    plot_figure.circle(
        x="x",
        y="y",
        source=data_source,
        color='color',
        size=6.5,
        alpha="alpha",
)

# invisible grid and axes
plot_figure.grid.visible = False
plot_figure.axis.visible = False
# tool bar location
plot_figure.toolbar_location = 'above'
# colored outline
plot_figure.outline_line_width = 7
plot_figure.outline_line_alpha = 0.3
plot_figure.outline_line_color = "#dc143c"

# don't remove dots when moving around to control lag (should be ok <10k glyphs)
plot_figure.lod_threshold = None

################ TASTE BUTTONS
fam = wines.groupby('Specialty').mean()
# wines.drop(['Alcohol', 'wine', 'Variety', 'Grapes', 'Specialty'], axis=1, inplace=True)
taste_threshold = 1

LABELS = fam.columns[(fam > taste_threshold).apply(any, 0)].to_list()

taste_buttons = CheckboxButtonGroup(labels=LABELS, active=[])

taste_buttons.js_on_click(CustomJS(
args=dict(
    source=buttons_data_source,
    matching_alpha=1,
    non_matching_alpha=1 - 0.95,
    reference = reference_data_by_fam,
    columns = LABELS,
), code= """
var activeButtons = this.active;
var single_wine_specialties = source.data['Specialty']
var chosenColumns = new Array()
let matches = []

activeButtons.forEach(element => chosenColumns.push(columns[element]))
let counter = 0

chosenColumns.forEach(element => {
counter += 1

console.log('printing element in' + element)
  for (var i = 0; i<reference.data[element].length; i+=1){
  if(parseFloat(reference.data[element][i],10) > 1){
    matches.forEach(match => {
    matches.push(reference.data[element].indexOf(reference.data[element][i]))
    })
    
  }
  }
})

console.log(matches)
    
    
""")
)

legendList = []
for k in range(len(unique_labels)):
    legendList.append((unique_labels[k],[plot_figure.renderers[k]]))

legend1 = Legend(items=list(map(legendList.__getitem__, range(0,6))),
                location=(5,-10),
                label_width=-5,
                title='Wine families',
                orientation='horizontal',
                label_standoff=1,
                spacing=5)

legend2 = Legend(items=list(map(legendList.__getitem__, range(6, 11))),
                location=(5,10),
                label_width=-5,
                orientation='horizontal',
                label_standoff=1,
                spacing=5)

plot_figure.add_layout(legend1, 'below')
plot_figure.add_layout(legend2, 'below')

plot_figure_widgets = row(plot_figure, taste_buttons)

show(plot_figure_widgets)
