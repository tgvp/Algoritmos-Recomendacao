import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt

def plot_rating_distribution(data: pd.DataFrame, plot_title: str = '') -> None:
    """Plots the rating distribution. Just ratings above 0 will be plotted.

    Args:
        data (pd.DataFrame): Data to plot.
        title (str, optional): Plot title: Ratings Distribution {plot_title}. Defaults to ''.
    Returns:
        None
    """
    fig = px.histogram(data[data.rating > 0], x='rating', color='rating', title=f'Ratings distribution {plot_title}')
    fig.update_layout(bargap=0.1, title_x=0.5)
    fig.update_layout(annotations=[
        dict(
            x=rating,
            y=count,
            text=f'{count:,}',
            xanchor='center',
            yanchor='bottom',
            showarrow=False,
            font=dict(
                size=12
            )
        ) for rating, count in zip(data[data.rating > 0].rating.value_counts().index,
                                data[data.rating > 0].rating.value_counts().values)
    ])
    fig.show()