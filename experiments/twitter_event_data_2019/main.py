# Created by Hansi at 7/1/2021
import os

from algo.embed2group import full_flow
from algo.embedding.model_learner import set_seed
from algo.utils.file_utils import delete_create_folder
from experiments.twitter_event_data_2019.args import DATA_FOLDER_PATH, TEMP_FOLDER, args, we_args, RANDOM_SEED

if __name__ == '__main__':
    set_seed(RANDOM_SEED)
    data_file_path = os.path.join(DATA_FOLDER_PATH, 'munliv-15.28-17.23.tsv')
    file_name = os.path.splitext(os.path.basename(data_file_path))[0]
    output_folder_path = os.path.join(TEMP_FOLDER, file_name)
    delete_create_folder(output_folder_path)

    full_flow(data_file_path, args, we_args, output_folder_path)
