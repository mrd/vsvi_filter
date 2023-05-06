# Simple program to load an image, run a semantic segmentation model on it
# (from MXNet model zoo) and save the results in a compressed numpy file.
import argparse
import numpy as np
from pathlib import Path
import sys
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform

parser = argparse.ArgumentParser(prog='run_segm.py', description='Run semantic segmentation')
parser.add_argument('filename', metavar='FILENAME', help='Image filename to process')
parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Run in verbose mode')
parser.add_argument('--modelname', metavar='MODEL', help='Use a specified model (from gluoncv.model_zoo)',default='psp_resnet101_citys')
parser.add_argument('--gpu', '-G', metavar='N', nargs='?', default=None, const=True, help='Use GPU (optionally specify which one)')
parser.add_argument('--outputfile', '-o', metavar='FILENAME', default=None, help='Output filename (default is input filename changed to have ".npz" suffix)')


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


    vlog(f'Loading image "{args.filename}".')
    img = image.imread(args.filename)

    img = test_transform(img, ctx)

    vlog(f'Loading model "{args.modelname}".')
    model = gluoncv.model_zoo.get_model(args.modelname, pretrained=True)

    vlog('Running model...')
    output = model.predict(img)

    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

    if args.outputfile is None:
        outputfile = Path(args.filename).with_suffix('.npz')
    else:
        outputfile = Path(outputfile)

    vlog(f'Saving predictions (shape={predict.shape}) into "{outputfile}".')
    np.savez_compressed(str(outputfile), predict=predict, modelname=args.modelname)

if __name__=='__main__':
    main()

# vim: ai sw=4 sts=4 ts=4 et
