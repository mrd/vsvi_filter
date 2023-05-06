# vsvi filter

Tools to help filter 'volunteered street view imagery' such as photos found on mapillary and similar sources.

## Installation

Have a recent version of Python3 and with pip: `pip install -r requirements.txt`.

## `run_segm.py`
### Simple program to run semantic segmentation on an image
#### Usage 
    
     run_segm.py [-h] [--verbose] [--modelname MODEL] [--gpu [N]]
                 [--outputfile FILENAME]
                 FILENAME

    positional arguments:
      FILENAME              Image filename to process

    options:
      -h, --help            show this help message and exit
      --verbose, -v         Run in verbose mode
      --modelname MODEL     Use a specified model (from gluoncv.model_zoo)
      --gpu [N], -G [N]     Use GPU (optionally specify which one)
      --outputfile FILENAME, -o FILENAME
                            Output filename (default is input filename changed to
                            have ".npz" suffix)
