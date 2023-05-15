#!/usr/bin/env python3
# Simple program to load an image, run a semantic segmentation model on it
# (from MXNet model zoo) and save the results in a compressed numpy file.
#
# See README.md.
# Licence:
#   Copyright 2023 Matthew Danish
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
import argparse
from time import time
import numpy as np
from pathlib import Path
import sys
import re
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform

parser = argparse.ArgumentParser(prog='run_segm.py', description='Run semantic segmentation')
parser.add_argument('paths', metavar='PATH', nargs='+', help='Image filenames or directories to process')
parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Run in verbose mode')
parser.add_argument('--output-extension', metavar='EXT', default='npz', help='Output filename extension (default: npz)')
parser.add_argument('--recursive', '-r', default=False, action='store_true',help='Recursively search for images in the given directory and subdirectories.')
parser.add_argument('--image-extensions', '-e', metavar='EXT', nargs='+', default=['jpg', 'jpeg'], help='Image filename extensions to consider (default: jpg jpeg). Case-insensitive.')
parser.add_argument('--no-detect-panoramic', default=False, action='store_true',help='Do not try to detect and correct panoramic images')
parser.add_argument('--scaledown-factor', '-s', default=4, type=float, help='Image scaling down factor')
parser.add_argument('--scaledown-interp', default=3, type=int, help='Interpolation method (see mxnet.image.imresize docs)')
parser.add_argument('--overwrite', '-O', action='store_true', default=False, help='Overwrite any existing output file')
parser.add_argument('--dry-run', action='store_true', default=False, help='Do not actually write any output file')
parser.add_argument('--modelname', metavar='MODEL', help='Use a specified model (from gluoncv.model_zoo)',default='psp_resnet101_citys')
parser.add_argument('--gpu', '-G', metavar='N', nargs='?', default=None, const=True, help='Use GPU (optionally specify which one)')
parser.add_argument('--exclusion-pattern', '-E', metavar='REGEX', default='.*(npz|mask|out|_x[0-9]+).*', help='Regex to indicate which files should be excluded from processing.')

def main():
    args = parser.parse_args()
    def vlog(s):
        if args.verbose:
            print(s)

    if args.gpu is not None:
        vlog(f'Using GPU ({args.gpu}).')
        if type(args.gpu)=='str':
            ctx = mx.gpu(int(args.gpu))
        elif type(args.gpu)=='int':
            ctx = mx.gpu(args.gpu)
        else:
            ctx = mx.gpu()
    else:
        vlog('Using CPU.')
        ctx = mx.cpu()

    vlog(f'Loading model "{args.modelname}".')
    model = gluoncv.model_zoo.get_model(args.modelname, pretrained=True, ctx=ctx)

    def do_file(inputpath):
        inputfilename = inputpath.name
        outputpath = inputpath.with_suffix(f'.{args.output_extension}')
        if outputpath.exists() and not args.overwrite:
            try:
                with np.load(outputpath) as f:
                    if 'predict' in f:
                        vlog(f'Skipping existing output file "{outputpath}".')
                        return
            except:
                pass
        vlog(f'Loading image "{inputpath}".')
        img = image.imread(inputpath)

        vlog(f'Image size={img.shape[1]}x{img.shape[0]}.')
        if not args.no_detect_panoramic and img.shape[1] >= img.shape[0]*2:
            img = image.fixed_crop(img, 0, 0, img.shape[1], img.shape[0]*3//4)
            vlog(f'Assuming panoramic image, cropping to {img.shape[1]}x{img.shape[0]}.')

        if args.scaledown_factor != 1:
            img = image.imresize(img,
                                 int(img.shape[1]//args.scaledown_factor),
                                 int(img.shape[0]//args.scaledown_factor),
                                 interp=int(args.scaledown_interp))
            vlog(f'Scaling down image to {img.shape[1]}x{img.shape[0]}.')

        img = test_transform(img, ctx)

        vlog('Running model...')
        t1 = time()

        output = model.predict(img)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

        t2 = time()
        vlog(f'Complete. Runtime: {(t2-t1):.2f}s.')

        if not args.dry_run:
            vlog(f'Saving predictions (shape={predict.shape}) into "{outputpath}".')
            np.savez_compressed(str(outputpath), predict=predict, modelname=args.modelname)

    image_extensions = [ e.lower() for e in args.image_extensions ]
    exclude = re.compile(args.exclusion_pattern)
    def recur(paths):
        for name in paths:
            p = Path(name)
            if p.is_file() and p.suffix.lower()[1:] in image_extensions:
                if not exclude.match(p.name):
                    do_file(p)
            elif args.recursive and p.is_dir():
                recur(p.iterdir())
    recur(args.paths)

if __name__=='__main__':
    main()

# vim: ai sw=4 sts=4 ts=4 et
