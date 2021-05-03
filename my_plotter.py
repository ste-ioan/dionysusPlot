from os import listdir
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from umap import UMAP
from bokeh.models import CustomJS, TextInput
from bokeh.models import Legend, LegendItem, LinearAxis
from bokeh.core.enums import LegendLocation
from bokeh.layouts import column
from bokeh.plotting import show

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

data_folder = 'C:\\Users\\sadek\\PycharmProjects\\wineScraper\\data\\'
files = [f for f in listdir(data_folder) if f.endswith('.csv')]

wines = []
for filename in files:
    wines.append(pd.read_csv(data_folder + '\\' + filename))

wines = pd.concat(wines)
wines.reset_index(inplace=True, drop=True)

# find the rare taste features (in less than 5 percent of the sample), drop them from the df
rare_features = wines.isnull().sum(axis=0)[wines.isnull().sum(axis=0) > (wines.shape[0]*0.95)].index.to_list()

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
wines[wines.columns[res[0]+1:].to_list()].max(axis=1)
# the mean is skewed but the median might be an acceptable cutoff, if a wine has less than
# that number of comments in its highest scoring feature, then we'll filter it out
# filtro = wines[wines.columns[res[0]+1:].to_list()].max(axis=1).median()

# no e' meglio per ora un filtro arbitrario (la feature piu' votata deve avere almeno x voti)
filtro = 30
wines.drop(wines[wines[wines.columns[res[0] + 1:].to_list()].max(axis=1) < filtro].index, inplace=True)

# stash the taste features, normalise and reinput them into the matrix, fill na's with 0s
taste_features_columns = wines.columns[res[0]+1:].to_list()
normalised_tastes = (wines[taste_features_columns] - wines[taste_features_columns].mean()) / wines[taste_features_columns].std()
wines[taste_features_columns] = normalised_tastes.fillna(0)
# give avg alcohol content to wines that are missing this data
wines['Alcohol'].fillna(wines['Alcohol'].mean().round(1), inplace=True)

# remove any rows that still have nas (no slider data for those wines probably)
wines.dropna(inplace=True)
wines.reset_index(inplace=True, drop=True)

hover_data = pd.DataFrame({'name': wines['wine'],
                           'price': wines['Price'],
                           'grapes': wines['Grapes']})

wines.drop(['Price', 'Rating'], axis=1, inplace=True)

# print out some info
print("There are {} wines and {} taste features in the dataset".format(wines.shape[0], wines.shape[1]))

# another (faster) way of doing things
mapper = umap.UMAP().fit(wines[wines.columns[3:].to_list()])

points = emb_get_embedding(mapper)

if subset_points is not None:
    if len(subset_points) != points.shape[0]:
        raise ValueError(
            "Size of subset points ({}) does not match number of input points ({})".format(
                len(subset_points), points.shape[0]
            )
        )
    points = points[subset_points]

if points.shape[1] != 2:
    raise ValueError("Plotting is currently only implemented for 2D embeddings")

if point_size is None:
    point_size = 100.0 / np.sqrt(points.shape[0])

data = pd.DataFrame(_get_embedding(umap_object), columns=("x", "y"))

if labels is not None:
    data["label"] = labels

    if color_key is None:
        unique_labels = np.unique(labels)
        num_labels = unique_labels.shape[0]
        color_key = _to_hex(
            plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
        )

    if isinstance(color_key, dict):
        data["color"] = pd.Series(labels).map(color_key)
    else:
        unique_labels = np.unique(labels)
        if len(color_key) < unique_labels.shape[0]:
            raise ValueError(
                "Color key must have enough colors for the number of labels"
            )

        new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
        data["color"] = pd.Series(labels).map(new_color_key)

    colors = "color"

elif values is not None:
    data["value"] = values
    palette = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, 256)))
    colors = btr.linear_cmap(
        "value", palette, low=np.min(values), high=np.max(values)
    )

else:
    colors = matplotlib.colors.rgb2hex(plt.get_cmap(cmap)(0.5))

if subset_points is not None:
    data = data[subset_points]
    if hover_data is not None:
        hover_data = hover_data[subset_points]

if points.shape[0] <= width * height // 10:

    if hover_data is not None:
        tooltip_dict = {}
        for col_name in hover_data:
            data[col_name] = hover_data[col_name]
            tooltip_dict[col_name] = "@{" + col_name + "}"
        tooltips = list(tooltip_dict.items())
    else:
        tooltips = None

    data["alpha"] = 1

    ## had to comment here to do iterate this by label to have legend STEMPI0
    data_source = bpl.ColumnDataSource(data)

    plot = bpl.figure(
        width=width,
        height=height,
        tooltips=tooltips,
        background_fill_color=background,
    )
    '''
    # iterating for legends here, below is back up of code STEMPI0
    for cluster in data['label'].unique():
        data_source = bpl.ColumnDataSource(data[data['label'] == cluster])

        plot.circle(
            x="x",
            y="y",
            source=data_source,
            color=colors,
            size=point_size,
            alpha="alpha",
            legend_label=cluster
        )

    '''
    plot.circle(
        x="x",
        y="y",
        source=data_source,
        color=colors,
        size=point_size,
        alpha="alpha",
    )

    plot.grid.visible = False
    plot.axis.visible = False

    if interactive_text_search:
        text_input = TextInput(value="", title="Search:")

        if interactive_text_search_columns is None:
            interactive_text_search_columns = []
            if hover_data is not None:
                interactive_text_search_columns.extend(hover_data.columns)
            if labels is not None:
                interactive_text_search_columns.append("label")

        if len(interactive_text_search_columns) == 0:
            warn(
                "interactive_text_search_columns set to True, but no hover_data or labels provided."
                "Please provide hover_data or labels to use interactive text search."
            )

        else:
            callback = CustomJS(
                args=dict(
                    source=data_source,
                    matching_alpha=interactive_text_search_alpha_contrast,
                    non_matching_alpha=1 - interactive_text_search_alpha_contrast,
                    search_columns=interactive_text_search_columns,
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

            plot = column(text_input, plot)

    # bpl.show(plot)
else:
    if hover_data is not None:
        warn(
            "Too many points for hover data -- tooltips will not"
            "be displayed. Sorry; try subssampling your data."
        )
    if interactive_text_search:
        warn(
            "Too many points for text search." "Sorry; try subssampling your data."
        )
    hv.extension("bokeh")
    hv.output(size=300)
    hv.opts('RGB [bgcolor="{}", xaxis=None, yaxis=None]'.format(background))
    if labels is not None:
        point_plot = hv.Points(data, kdims=["x", "y"])
        plot = hd.datashade(
            point_plot,
            aggregator=ds.count_cat("color"),
            color_key=color_key,
            cmap=plt.get_cmap(cmap),
            width=width,
            height=height,
        )
    elif values is not None:
        min_val = data.values.min()
        val_range = data.values.max() - min_val
        data["val_cat"] = pd.Categorical(
            (data.values - min_val) // (val_range // 256)
        )
        point_plot = hv.Points(data, kdims=["x", "y"], vdims=["val_cat"])
        plot = hd.datashade(
            point_plot,
            aggregator=ds.count_cat("val_cat"),
            cmap=plt.get_cmap(cmap),
            width=width,
            height=height,
        )
    else:
        point_plot = hv.Points(data, kdims=["x", "y"])
        plot = hd.datashade(
            point_plot,
            aggregator=ds.count(),
            cmap=plt.get_cmap(cmap),
            width=width,
            height=height,
        )
