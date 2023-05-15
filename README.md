# vsvi filter

Tools to help filter 'volunteered street view imagery' such as photos found on mapillary and similar sources.

## Set-up

Have a recent version of Python3 and with pip: `pip install -r requirements.txt`.

## `run_segm.py`

Simple program to run semantic segmentation on an image

### Usage

    run_segm.py [-h] [--verbose] [--output-extension EXT] [--recursive]
                     [--image-extensions EXT [EXT ...]] [--no-detect-panoramic]
                     [--scaledown-factor SCALEDOWN_FACTOR]
                     [--scaledown-interp SCALEDOWN_INTERP] [--overwrite]
                     [--dry-run] [--modelname MODEL] [--gpu [N]]
                     [--exclusion-pattern REGEX]
                     PATH [PATH ...]

    Run semantic segmentation

    positional arguments:
      PATH                  Image filenames or directories to process

    optional arguments:
      -h, --help            show this help message and exit
      --verbose, -v         Run in verbose mode
      --output-extension EXT
                            Output filename extension (default: npz)
      --recursive, -r       Recursively search for images in the given directory
                            and subdirectories.
      --image-extensions EXT [EXT ...], -e EXT [EXT ...]
                            Image filename extensions to consider (default: jpg
                            jpeg). Case-insensitive.
      --no-detect-panoramic
                            Do not try to detect and correct panoramic images
      --scaledown-factor SCALEDOWN_FACTOR, -s SCALEDOWN_FACTOR
                            Image scaling down factor
      --scaledown-interp SCALEDOWN_INTERP
                            Interpolation method (see mxnet.image.imresize docs)
      --overwrite, -O       Overwrite any existing output file
      --dry-run             Do not actually write any output file
      --modelname MODEL     Use a specified model (from gluoncv.model_zoo)
      --gpu [N], -G [N]     Use GPU (optionally specify which one)
      --exclusion-pattern REGEX, -E REGEX
                            Regex to indicate which files should be excluded from
                            processing.
