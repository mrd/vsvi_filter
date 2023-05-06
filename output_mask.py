#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import sys
from scipy.signal import find_peaks
from PIL import ImageDraw
from gluoncv.utils.viz import get_color_pallete
import cv2

parser = argparse.ArgumentParser(prog='output_mask.py', description='Output image mask with possible road centres marked')
parser.add_argument('filename', metavar='FILENAME', help='Saved numpy (.npz or .npy) file to process')
parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Run in verbose mode')
parser.add_argument('--blur', action='store_true', default=False, help='Run Gaussian blur before finding edges')
parser.add_argument('--maskfile', '-m', metavar='FILENAME', default=None, help='Output filename for mask image')
parser.add_argument('--edgefile', '-e', metavar='FILENAME', default=None, help='Output filename for edges image')
parser.add_argument('--linefile', '-l', metavar='FILENAME', default=None, help='Output filename for lines image')
parser.add_argument('--blurfile', '-B', metavar='FILENAME', default=None, help='Output filename for Gaussian blurred image')
parser.add_argument('--blobfile', '-b', metavar='FILENAME', default=None, help='Output filename for blobs image')
parser.add_argument('--dataset', metavar='DATASET', default=None, help='Override segmentation dataset name (for visualisation)')
parser.add_argument('--road-peaks-distance', metavar='N', default=None, type=int, help='Distance between peaks of road pixels')
parser.add_argument('--road-peaks-prominence', metavar='N', default=None, type=int, help='Prominence of peaks of road pixels')
parser.add_argument('--houghlines-rho', metavar='RHO', default=None, type=float, help='Hough transform RHO parameter')
parser.add_argument('--houghlines-theta', metavar='THETA', default=None, type=float, help='Hough transform THETA parameter')
parser.add_argument('--houghlines-threshold', metavar='THRESH', default=None, type=int, help='Hough transform THRESHOLD parameter')
parser.add_argument('--houghlines-min-theta', metavar='THETA', default=None, type=float, help='Hough transform MIN_THETA parameter')
parser.add_argument('--houghlines-max-theta', metavar='THETA', default=None, type=float, help='Hough transform MAX_THETA parameter')


# https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])

def road_pixels_per_col(pred):
    a = pred == 0.0
    out = np.zeros(a.shape[1])
    for i in range(a.shape[1]):
      (z, p, v) = rle(a[:,i])
      out[i]=z[v.nonzero()].max(initial=0)
    return out

def road_centres(pred, distance=2000, prominence=100):
    road = road_pixels_per_col(pred)
    return find_peaks(road,distance=distance,prominence=prominence)[0]

def main():
    args = parser.parse_args()
    def vlog(s):
        if args.verbose:
            print(s)

    filename = args.filename

    vlog(f'Loading "{filename}".')
    if Path(filename).suffix == '.npz':
        with np.load(filename) as f:
            predict = f['predict']
            modelname = str(f['modelname'])
    else:
        predict = np.load(filename)
        modelname = None

    if predict.shape[1] >= predict.shape[0] * 2:
        # panoramic
        predictplus = np.append(predict, predict[:,:predict.shape[1]//4],axis=1)
        vlog(f'Assuming panoramic input, extending width to {predictplus.shape[1]}.')
    else:
        predictplus = predict


    distance=args.road_peaks_distance
    if distance is None:
        distance = int(2000 * predict.shape[1] // 5760)

    prominence=args.road_peaks_prominence
    if prominence is None:
        prominence = int(100 * predict.shape[0] // 2880)

    vlog(f'Seeking road centres (using pixel segmentation; distance={distance}, prominence={prominence})...')

    centres=road_centres(predictplus, distance=distance, prominence=prominence)
    vlog(f'Found road centres: {centres}.')
    dataset = args.dataset or 'citys'
    if modelname is not None and args.dataset is None:
        dataset = modelname.split('_')[-1] 

    vlog(f'Generating mask image (using dataset colouring "{dataset}").')
    mask = get_color_pallete(predictplus, dataset)

    #img = cv2.imread(str(jpgfilename))
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.array(mask.convert('L'))

    
    if args.blur or args.blurfile:
        k=5
        vlog(f'Running Gaussian blur with kernel {k}x{k}.')
        blur = cv2.GaussianBlur(gray, (k, k), 1)
        if args.blurfile:
            vlog(f'Writing blur image "{args.blurfile}".')
            cv2.imwrite(args.blurfile,blur)
    else:
        blur = None

    edgeImg = cv2.Canny(blur if blur is not None else gray, 40, 255)
    if args.edgefile:
        vlog(f'Writing edges image to "{args.edgefile}".')
        cv2.imwrite(args.edgefile, edgeImg)

    #lines = cv2.HoughLinesP(edgeImg, 1, np.pi / 180, 50, None, 50, 10)
    rho = float(args.houghlines_rho or 1)
    theta = float(args.houghlines_theta or np.pi/120)
    threshold = int(args.houghlines_threshold or 120)
    min_theta = float(args.houghlines_min_theta or np.pi/36)
    max_theta = float(args.houghlines_max_theta or np.pi-np.pi/36)
    vlog(f'Running HoughLines (rho={rho}, theta={theta}, threshold={threshold}, min_theta={min_theta}, max_theta={max_theta}).')
    lines = cv2.HoughLines(edgeImg, rho, theta, threshold, min_theta=min_theta, max_theta=max_theta)

    cdst = cv2.cvtColor(edgeImg, cv2.COLOR_GRAY2BGR)

    # https://stackoverflow.com/questions/57535865/extract-vanishing-point-from-lines-with-open-cv
    if lines is not None:
        vlog(f'Line count: {len(lines)}')
        for line in lines:
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 10000*(-b))
            y1 = int(y0 + 10000*(a))
            x2 = int(x0 - 10000*(-b))
            y2 = int(y0 - 10000*(a))
            cv2.line(cdst,(x1,y1),(x2,y2),(0,255,0),1, cv2.LINE_AA)

    if args.linefile:
        vlog(f'Writing lines image to "{args.linefile}".')
        cv2.imwrite(args.linefile, cdst)

    blobs = np.copy(cdst)

    kernel = np.ones((3,3),np.uint8)
    blobs = cv2.erode(blobs,kernel,iterations=1)
    kernel = np.ones((9,9),np.uint8)
    blobs = cv2.dilate(blobs,kernel,iterations=1)
    kernel = np.ones((11,11),np.uint8)
    blobs = cv2.erode(blobs,kernel,iterations=1)
    blobs = cv2.dilate(blobs,kernel,iterations=1)
    if args.blobfile:
        vlog(f'Writing blobs image to "{args.blobfile}".')
        cv2.imwrite(args.blobfile, blobs)

    grayblobs = cv2.cvtColor(blobs,cv2.COLOR_BGR2GRAY)
    grayblobs1d = np.count_nonzero(grayblobs,axis=0)
    #blobvp1 = grayblobs1d.argmax()
    #blobvp2 = (blobvp1 + predict.shape[1]//2) % predict.shape[1]
    #print(f'blobvp1={blobvp1} blobvp2={blobvp2}')

    blobvps = find_peaks(grayblobs1d, distance=predict.shape[1]//4)[0]
    vlog(f'Found vanishing points (using coalesced blobs): {blobvps}.')


    mask = mask.convert('RGB')
    draw = ImageDraw.Draw(mask)
    for b1 in blobvps:
        draw.line((b1,0,b1,mask.size[1]),width=3,fill=(256,0,0)) 
        b2 = (b1 + predict.shape[1]//2) % predict.shape[1]
        draw.line((b2,0,b2,mask.size[1]),width=1,fill=(128,0,0)) 
    for c1 in centres:
        draw.line((c1,0,c1,mask.size[1]),width=3,fill=(0,256,0))
        c2 = (c1 + predict.shape[1]//2) % predict.shape[1]
        draw.line((c2,0,c2,mask.size[1]),width=1,fill=(0,256,0))
        #blobDistThreshold = 100
        #if abs(blobvp1 - c1) < blobDistThreshold or abs(blobvp2 - c1) < blobDistThreshold or \
        #  abs(blobvp1 - c2) < blobDistThreshold or abs(blobvp2 - c2) < blobDistThreshold:
        #    print(f'c1={c1} c2={c2}') 
        #    draw.line((c2,0,c2,mask.size[1]),fill=128)
        #    break
    if args.maskfile:
        vlog(f'Writing mask image to "{args.maskfile}".')
        mask.save(args.maskfile)

if __name__=='__main__':
    main()

# vim: ai sw=4 sts=4 ts=4 et
