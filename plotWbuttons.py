from os import listdir
import sys
import pandas as pd
import seaborn as sns
import umap
from bokeh.models import CustomJS, TextInput, RangeSlider, HoverTool
from bokeh.models import Legend, LegendItem, LinearAxis, CategoricalColorMapper, ColumnDataSource, CheckboxGroup, CDSView, BooleanFilter
from bokeh.layouts import column, row
from bokeh.plotting import figure, show, output_file, save
from bokeh.palettes import Spectral10
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import product
from bokeh.resources import CDN
from bokeh.embed import components

def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


def make_specialty_fig(wine_frame):
    plot_data = wine_frame.filter(['Specialty', 'Rating'], axis=1).groupby('Specialty').mean()['Rating']
    plot_data = plot_data.sort_values(ascending=False)

    ax = sns.barplot(x=plot_data.index, y=plot_data.values)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6.9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
    ax.set_ylim(3, plot_data.max()+0.1)
    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig("output.png")


sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
data_folder = 'C:\\Users\\sadek\\PycharmProjects\\wineScraper\\data\\merged\\'

try:
    listdir(data_folder)

except:
    print("Could not find files directory...Trying with sys.args")
    data_folder = sys.argv[1]

# glob package is a valid alternative to this, try out the difference next time
# you wanna find files in folders
files = [f for f in listdir(data_folder) if f.endswith('.csv')]

wines = []
for filename in files:
    wines.append(pd.read_csv(data_folder + '\\' + filename))

wines = pd.concat(wines)
wines.reset_index(inplace=True, drop=True)

# sum some redundant features
wines['Anise'] = (wines.loc[:, 'Anise'].fillna(0) + wines.loc[:, 'Star anise'].fillna(0)).replace(
    {'0': np.nan, 0: np.nan})
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
wines['Citrus'] = (wines.loc[:, 'Citrus'].fillna(0) + wines.loc[:, 'Lemon'].fillna(0)).replace({'0': np.nan, 0: np.nan})

wines['Cherry'] = (wines.loc[:, 'Cherry'].fillna(0) + wines.loc[:, 'Black cherry'].fillna(0)).replace({'0': np.nan, 0: np.nan})

wines.drop(['Star anise', 'Orange peel', 'Blood orange', 'Campfire', 'Black fruit', 'Dried rose', 'Potpourri', 'Lemon',
            'Black cherry'],
           axis=1, inplace=True)

# find the rare taste features (in less than 15 percent of the sample), drop them from the df
rare_features = wines.isnull().sum(axis=0)[wines.isnull().sum(axis=0) > (wines.shape[0] * 0.85)].index.to_list()

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

# find index for first taste feature
res = [i for i, val in enumerate(wines.columns == 'Rating') if val]

# establish a cutoff to exclude wines with too few votes on the tastes
filtro = 30
wines.drop(wines[wines[wines.columns[res[0] + 1:].to_list()].fillna(0).max(axis=1) < filtro].index, inplace=True)

print("Votes from at least {} people were employed to estimate the taste features".format(wines.loc[:,wines.columns[res[0]+1:]].max(axis=1).sum()))

# stash the taste features, normalise and reinput them into the matrix, fill na's with 0s
taste_features_columns = wines.columns[res[0] + 1:].to_list()
normalised_tastes = wines[taste_features_columns].subtract(wines[taste_features_columns].mean(axis=1), axis=0).divide(
    wines[taste_features_columns].std(axis=1), axis=0)
wines[taste_features_columns] = normalised_tastes.fillna(0)

# let's normalise also the slider data
wines.loc[:,'Bold':'Soft'] = wines.loc[:,'Bold':'Soft'].subtract(wines.loc[:,'Bold':'Soft'].mean()).divide(
    wines.loc[:,'Bold':'Soft'].std())

# give avg alcohol content to wines that are missing this data, by specialty (this breaks down very separate clusters!)
# alternative is to give avg alcohol content, that brings it more together
for specialty in wines['Specialty'].unique():
    copy = wines.loc[wines.loc[:, 'Specialty'] == specialty, 'Alcohol']
    media = wines[wines['Specialty'] == specialty]['Alcohol'].mean()
    wines.loc[wines.loc[:, 'Specialty'] == specialty, 'Alcohol'] = copy.fillna(media)

# here normalise alcohol content
wines.loc[:, 'Alcohol'] = wines.loc[:, 'Alcohol'].subtract(wines.loc[:, 'Alcohol'].mean()).divide(
    wines.loc[:, 'Alcohol'].std())
# fix price
wines['Price'].fillna(0, inplace=True)

# make bar plot of wine specialties, before removing ratings
# let's include only specialties with a min number of bottles
specs_to_plot = wines.groupby("Specialty").count().index[wines.groupby("Specialty").count()['wine']>30]

make_specialty_fig(wines.loc[wines['Specialty'].isin(specs_to_plot),])
wines.drop(['Price','Rating'], axis=1, inplace=True)

# remove any rows that still have nas (no slider data for those wines probably)
wines.dropna(inplace=True)
wines.reset_index(inplace=True, drop=True)
# create the hover lists that appear when mouse glides over data points
hover_data = pd.DataFrame({'specialty': wines['Specialty'].str.lower(),
                           'grapes': wines['Grapes'].str.lower(),
                           })

# print out some info
print("There are {} wines and {} taste features in the dataset".format(wines.shape[0], wines.shape[1]))

# mapper parameters
fit = umap.UMAP(
    n_neighbors=18,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    transform_seed=42,
)

# fit the mapper
taste_data = wines.loc[:,'Bold':]
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
data['specialty'] = wines['Specialty']
# data["price"] = hover_data["price"]

tooltip_dict = {}
for col_name in hover_data:
    data[col_name] = hover_data[col_name]
    tooltip_dict[col_name] = "@{" + col_name + "}"
tooltips = list(tooltip_dict.items())

data_source = ColumnDataSource(data)
reference_data_by_fam = ColumnDataSource(wines.groupby('Specialty').mean())

print("Creating struct file...")
output_file('index.html')

######################################## CREATE PLOT ###################################################################
plot_figure = figure(
    title='Dionysos tastespace of {} wines'.format(wines.shape[0]),
    plot_width=1000,
    plot_height=700,
    tooltips=tooltips,
    background_fill_color='white',
)

for variety in unique_labels:
    plot_figure.circle(
        x="x",
        y="y",
        source=data_source,
        view=CDSView(source=data_source, filters=[BooleanFilter(data_source.data['label'] == variety)]),
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

########################################### TASTE BUTTONS ##############################################################
# I NEED TO CREATE ANOTHER TASTE BUTTON WIDGET SO LABELS CAN BE SPLIT AMONG THEM, OTHERWISE TOO LONG
# BOTH WIDGETS NEED TO BE AWARE OF THE CHOSEN COLUMNS IN BOTH AND ACT ACCORDINGLY.

fam = wines.groupby('Specialty').mean()
#fam.to_csv('specialtiesStats.csv')

# wines.drop(['Alcohol', 'wine', 'Variety', 'Grapes', 'Specialty'], axis=1, inplace=True)
taste_threshold = 1

LABELS = fam.columns[(fam > taste_threshold).apply(any, 0)].to_list()

for ugh in unwanted_labels:
    LABELS.remove(ugh)

taste_buttons = CheckboxGroup(labels=LABELS, active=[], height_policy='fixed',height=300)

taste_buttons.js_on_click(CustomJS(
args=dict(
    source=data_source,
    matching_alpha=1,
    non_matching_alpha=1 - 0.95,
    reference = reference_data_by_fam,
    columns = LABELS,
), code= """
var activeButtons = this.active;
var chosenColumns = new Array()
let matches = []
let specialtyIndex = []
let specialtyStrings = []

activeButtons.forEach(element => chosenColumns.push(columns[element]))

function onlyUnique(value, index, self) {
  return self.indexOf(value) === index;
}

function getOccurrence(array, value) {
    return array.filter((v) => (v === value)).length;
}

chosenColumns.forEach(element => {

for (var i = 0; i < reference.data[element].length; i += 1) {
    if (parseFloat(reference.data[element][i], 10) > 1) {
        matches.push(i);
    }
}

})

let numsInBothColumns = matches.filter(onlyUnique);

numsInBothColumns.forEach(
    (index) => {
        if(getOccurrence(matches, index) === chosenColumns.length) {
            specialtyIndex.push(index)
        }
    }
)

for (var i = 0; i < specialtyIndex.length; i += 1) {
        specialtyStrings.push(reference.data['Specialty'][specialtyIndex[i]].toLowerCase());
    }

if (chosenColumns.length > 0){
var string_match = false;
for (var i = 0; i < source.data['specialty'].length; i++) {
    string_match = false
    if (specialtyStrings.includes(String(source.data['specialty'][i]))) {
       string_match = true
    }
    if (string_match){
    source.data['alpha'][i] = matching_alpha
    }else{
    source.data['alpha'][i] = non_matching_alpha
    }
    }
    source.change.emit();
}else{    
if (chosenColumns.length === 0){
for (var i = 0; i < source.data['specialty'].length; i++){
source.data['alpha'][i] = matching_alpha
}
source.change.emit();  
}  
}
""")
)

############################################ LEGEND ####################################################################
legendList = []
for k in range(len(unique_labels)):
    legendList.append((unique_labels[k],[plot_figure.renderers[k]]))

legend1 = Legend(items=list(map(legendList.__getitem__, range(0,6))),
                location=(5,-10),
                label_width=-5,
                title='Wine families',
                orientation='horizontal',
                label_standoff=2,
                spacing=10)

legend2 = Legend(items=list(map(legendList.__getitem__, range(6, 11))),
                location=(5,10),
                label_width=-5,
                orientation='horizontal',
                label_standoff=2,
                spacing=10)

plot_figure.add_layout(legend1, 'below')
plot_figure.add_layout(legend2, 'below')

plot_figure_widgets = row(plot_figure, taste_buttons)

show(plot_figure_widgets)

# these should be copied inside the webpage html
script1, div1 = components(plot_figure_widgets)
# i'll need cdn_js[0] and cdn_js[1], same for cdn_css ([1] is for widgets
cdn_js = CDN.js_files
cdn_css = CDN.css_files