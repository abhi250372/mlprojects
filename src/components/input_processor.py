import json
from collections import defaultdict
import pandas as pd
from pathlib import Path

class InputProcessor:
    def __init__(self):
        self.data = None

    def load_data(self, file_path=None):
        self.data = pd.read_csv(file_path)
        return self.data