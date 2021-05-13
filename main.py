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


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
data_folder = 'C:\\Users\\sadek\\PycharmProjects\\wineScraper\\data\\merged\\'

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
wines['Anise'] = (wines.loc[:,'Anise'].fillna(0) + wines.loc[:,'Star anise'].fillna(0)).replace({'0':np.nan, 0:np.nan})
wines['Tropical'] = (wines.loc[:,'Tropical'].fillna(0) + wines.loc[:,'Mango'].fillna(0) + wines.loc[:,'Pineapple'].fillna(0) + wines.loc[:,'Green papaya'].fillna(0) +
                     wines.loc[:,'Passion fruit'].fillna(0) + wines.loc[:,'Guava'].fillna(0) + wines.loc[:,'Green mango'].fillna(0)).replace({'0':np.nan, 0:np.nan})
wines['Orange'] = (wines.loc[:,'Orange'].fillna(0) + wines.loc[:,'Blood orange'].fillna(0)).replace({'0':np.nan, 0:np.nan})
wines['Smoke'] = (wines.loc[:,'Smoke'].fillna(0) + wines.loc[:,'Campfire'].fillna(0)).replace({'0':np.nan, 0:np.nan})
wines['Orange zest'] = (wines.loc[:,'Orange zest'].fillna(0) + wines.loc[:,'Orange peel'].fillna(0) + wines.loc[:,'Orange rind'].fillna(0)).replace({'0':np.nan, 0:np.nan})
wines['Dark fruit'] = (wines.loc[:,'Dark fruit'].fillna(0) + wines.loc[:,'Black fruit'].fillna(0)).replace({'0':np.nan, 0:np.nan})
wines['Dried flowers'] =  (wines.loc[:,'Dried flowers'].fillna(0) + wines.loc[:,'Dried rose'].fillna(0) + wines.loc[:,'Potpourri'].fillna(0)).replace({'0':np.nan, 0:np.nan})

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

# establish a cutoff to exclude wines with too few votes on the tastes
filtro = 30
wines.drop(wines[wines[wines.columns[res[0] + 1:].to_list()].fillna(0).max(axis=1) < filtro].index, inplace=True)

# stash the taste features, normalise and reinput them into the matrix, fill na's with 0s
taste_features_columns = wines.columns[res[0] + 1:].to_list()
normalised_tastes = wines[taste_features_columns].subtract(wines[taste_features_columns].mean(axis=1), axis=0).divide(
    wines[taste_features_columns].std(axis=1), axis=0)
wines[taste_features_columns] = normalised_tastes.fillna(0)

# find indices for slider taste features
first = [i for i, val in enumerate(wines.columns == 'Bold') if val]
last = [i for i, val in enumerate(wines.columns == 'Soft') if val]

# let's normalise also the slider data (sweet, dry, etc)
wines[wines.columns[first[0]:last[0]]] = (wines[wines.columns[first[0]:last[0]]] - wines[wines.columns[first[0]:last[0]]].mean()) / wines[wines.columns[first[0]:last[0]]].std()

# give avg alcohol content to wines that are missing this data, by specialty
for specialty in wines['Specialty'].unique():
    copy = wines.loc[wines.loc[:, 'Specialty'] == specialty, 'Alcohol']
    media = wines[wines['Specialty'] == specialty]['Alcohol'].mean()
    wines.loc[wines.loc[:, 'Specialty'] == specialty, 'Alcohol'] = copy.fillna(media)

hover_data = pd.DataFrame({'name': wines['wine'].str.lower(),
                           'price': wines['Price'],
                           'grapes': wines['Grapes'].str.lower()})

wines.drop(['Price', 'Rating'], axis=1, inplace=True)
# remove any rows that still have nas (no slider data for those wines probably), after fixing price
wines.dropna(inplace=True)
wines.reset_index(inplace=True, drop=True)

# print out some info
print("There are {} wines and {} taste features in the dataset".format(wines.shape[0], wines.shape[1]))

# fit the mapper
mapper = umap.UMAP().fit(wines[wines.columns[3:].to_list()])
# manipulate data for the plots, will probably have to do this for both Specialty and Variety
data = pd.DataFrame(mapper.embedding_, columns=("x", "y"))
data['label'] = wines['Variety']
unique_labels = np.unique(data['label'])
num_labels = unique_labels.shape[0]
color_key = _to_hex(plt.get_cmap('Spectral')(np.linspace(0, 1, num_labels)))
new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
data["color"] = pd.Series(wines['Variety']).map(new_color_key)
data["alpha"] = 1
data["wine_name"] = wines["wine"].str.lower()
data["price"] = hover_data["price"]

tooltip_dict = {}
for col_name in hover_data:
    data[col_name] = hover_data[col_name]
    tooltip_dict[col_name] = "@{" + col_name + "}"
tooltips = list(tooltip_dict.items())

data_source = ColumnDataSource(data)

print("Creating struct file...")
output_file('red_wines_struct_new2.html')

##### CREATE PLOT
plot_figure = figure(
    title='Dionysus tastespace of {} red wines - alpha'.format(wines.shape[0]),
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
plot_figure.toolbar_location = 'above'

########### PRICER SLIDER
price_slider = RangeSlider(start=hover_data['price'].min(), end=hover_data['price'].max(), value=(7,20), step=.1, title="Price")

price_slider_callback = CustomJS(
    args=dict(
    source=data_source,
    matching_alpha=1,
    non_matching_alpha=1 - 0.95,
    search_columns=["price"],
    ),code="""
    var price_data = source.data;
    var price_range = this.value;
    
    var search_columns_dict = {}
    for (var col in search_columns){
        search_columns_dict[col] = search_columns[col]
    }
    var text_search = document.getElementsByName("wine_text_input")[0].value;
    var price_range_match = false;
    for (var i = 0; i < price_data.x.length; i++) {
        price_range_match = false
        for (var j in search_columns_dict) {
            if (price_data[search_columns_dict[j]][i] > price_range[0]
            && price_data[search_columns_dict[j]][i] < price_range[1]) {
                if (source.data["name"][i].includes(text_search)) {
                    price_range_match = true;
                }
            }
        }
        if (price_range_match){
            price_data['alpha'][i] = matching_alpha
        }else{
            price_data['alpha'][i] = non_matching_alpha
        }
    }
    source.change.emit();
""")

price_slider.js_on_change("value", price_slider_callback)

################ TEXT INPUT
# Search bar text input
text_input = TextInput(name="wine_text_input", value="", title="Search for wine:")

# This is executed once a given event is triggered. This event is declared using model.js_on_event('<event_type>', callback) <-- even type can be tap, click etc
callback = CustomJS(
args=dict(
    source=data_source,
    matching_alpha=1,
    non_matching_alpha=1 - 0.95,
    search_columns=["wine_name"],
    price_slider_value=price_slider,
),
code="""
var data = source.data;
var price_data = data["price"]
var text_search = cb_obj.value;
var price_range = price_slider_value["value"];
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
            if (price_data[i] > price_range[0]
            && price_data[i] < price_range[1]) {
                string_match = true

            }
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

################ TASTE BUTTONS
wines.to_csv('normalised_wines')
wines.drop('Alcohol', 'wine', 'Variety', 'Grapes',  axis=1, inplace=True)

taste_threshold = 1.5

LABELS = wines.columns[(wines > taste_threshold).apply(any, 0)].to_list()

taste_buttons = CheckboxButtonGroup(labels=LABELS, active=[0, 1])

callback = CustomJS(
args=dict(
    source=data_source, #this does not have info needed for buttons to work atm
    matching_alpha=1,
    non_matching_alpha=1 - 0.95,
    search_columns=["wine_name"], #this does not have info needed for buttons to work atm
    price_slider_value=taste_buttons,# probably needs active here instead of value?
),
code="""
var data = source.data;
var price_data = data["price"]
var text_search = cb_obj.value;
var price_range = price_slider_value["value"];
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
            if (price_data[i] > price_range[0]
            && price_data[i] < price_range[1]) {
                string_match = true

            }
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

taste_buttons.js_on_click("value", callback)


########### WRAP IT UP n PLOT
# have to put a row(plot_figure,buttons) inside this column call
plot_figure = column(text_input, row(plot_figure, taste_buttons), price_slider)

show(plot_figure)
