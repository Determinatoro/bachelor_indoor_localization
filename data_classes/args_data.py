from dataclasses import dataclass

import numpy as np


@dataclass
class ArgsData:
    def __init__(self):
        self.site_ids = []

    root_data_path: str
    site_ids: list
