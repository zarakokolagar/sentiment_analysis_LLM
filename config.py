import os

# Define the base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the data directory
DATA_DIR = os.path.join(BASE_DIR, 'data')

#Define predictions directory
PRED_DIR = os.path.join(BASE_DIR, 'predictions')

#define plot directory for predictions
PLOT_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

SAMPLED_DATA_PATH = os.path.join(DATA_DIR,'sampled_dataset.tsv')
