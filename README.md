# vsvi filter

Tools to help filter 'volunteered street view imagery' such as photos found on mapillary and similar sources.

## Set-up

Have a recent version of Python3 and with pip: `pip install -r requirements.txt`.

## `run_segm.py`

Simple program to run semantic segmentation on images and output the result
into accompanying '.npz' files for further processing.

### Examples

* Run with the first GPU on `image1.jpg`, in verbose mode:
  - `./run_segm.py --gpu 0 -v image1.jpg`

* Run on `image1.jpg` and recursively on directory `dir_of_jpgs`:
  - `./run_segm.py -v -r image1.jpg dir_of_jpgs/`

* Run recursively on `dir_of_pngs` and `dir_of_jpgs` looking for PNG and JPG files:
* `./run_segm.py -v -e png jpeg jpg -r dir_of_pngs dir_of_jpgs/`

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
