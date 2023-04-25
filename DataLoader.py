import pandas as pd
from IPython.display import display, HTML


class Loader:

    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def show_sample(self):
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        to_show = self.data.sample(50)
        display(to_show)

