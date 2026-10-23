import os
from src.EDA import singular_attribute_EDA


def create_folder(folder_name):
    """Create the output folder if it does not exist."""
    os.makedirs(folder_name, exist_ok=True)


def singular_attribute_EDA_all(data, singular_attribute_EDA_input, save_folder):
    """Plot every column in singular_attribute_EDA_input into save_folder (already created)."""
    for key in singular_attribute_EDA_input:
        singular_attribute_EDA(data[key], singular_attribute_EDA_input[key], save_folder)
