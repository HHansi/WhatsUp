# WhatsUp

WhatsUp is a a event resolution method for co-occurring events in social media data. 

More details will be available with the paper "WhatsUp: An Event Resolution Approach for Co-occurring Events in Social 
Media" which is under the review process currently. 

## About
This is a Python 3.7 implementation of WhatsUp. <br>
All the used packages are listed in [requirements.txt](https://github.com/HHansi/WhatsUp/blob/master/requirements.txt).

## Run the Code
Clone the repository and install the libraries using the following command (preferably inside a conda environment).
```
pip install -r requirements.txt
```
Also, make sure to download the necessary NLTK packages and set variables as mentioned in [requirements.txt](https://github.com/HHansi/WhatsUp/blob/master/requirements.txt).

Run WhatsUp full flow (full_flow function in [whatsup.py](https://github.com/HHansi/WhatsUp/blob/master/algo/whatsup.py)) given the following inputs.
* data_file_path (str): path to input data file
* args (JSON object): arguments for event detection <br>
A sample JSON created for Twitter-Event-Data-2019 is available in [args.py](https://github.com/HHansi/WhatsUp/blob/master/experiments/twitter_event_data_2019/args.py), and more details about the parameter selection can be found from the paper.
* we_args (JSON object): arguments to learn word embeddings <br>
A sample JSON created for Twitter-Event-Data-2019 is available in [args.py](https://github.com/HHansi/WhatsUp/blob/master/experiments/twitter_event_data_2019/args.py), and more details about the parameter selection can be found from the paper.
* output_folder_path (str): folder path to save outputs

### Input data file format
Input data file is a .tsv file formatted as follows:

* should contain a post (e.g. tweet) per line
* should contain 3 compulsory columns with headers: id, timestamp and text (any other column is ignored)
* timestamp should be formatted as %Y-%m-%d %H:%M:%S (e.g. 2019-10-20 15:25:00)

### Output folder format

Output folder will contain the following files and folders.

* cleaned.tsv - cleaned version of the input data file
* time-windows - folder of .tsv files/data per separate time windows
* word-embedding - folder of learned embedding models per time window
* stat - folder of extracted statistical information per time window
* results - folder of detected events with event windows

## Experiments
WhatsUp has experimented on [Twitter-Event-Data-2019](https://github.com/HHansi/Twitter-Event-Data-2019). Please refer to 
our paper for more details about the experiments and results. The codes used for experiments are available in 
[experiments](https://github.com/HHansi/WhatsUp/tree/master/experiments) folder.

None of the used data files is available in this repository due to the restrictions of [Twitter Developer Policy](https://developer.twitter.com/en/developer-terms/agreement-and-policy).
Please refer to the original data repository mentioned above to get access to the data.


