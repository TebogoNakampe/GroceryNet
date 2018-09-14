#!/usr/bin/python3
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import glob
import time
import csv
import caffe


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--test_file",
        default="test/test.txt",
        help="Input image list file."
    )
    parser.add_argument(
        "--test_dir",
        default="./test",
        help="path to input image file."
    )
    parser.add_argument(
        "--output",
        default="result",
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir, "models/deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir, "models/caffenet_train_iter_30000.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir, 'data/imagenet_mean.binaryproto'),
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--synset_words",
        default='synset_words.txt',
        help="Object id and name list."
    )
    parser.add_argument(
        "--top_k",
        default='3',
        help="Number of candidate."
    )
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None

    if args.mean_file:
        print("MeanFile: %s\n" % args.mean_file)
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open(args.mean_file, 'rb').read()
        blob.ParseFromString(data)
        arr = np.array(caffe.io.blobproto_to_array(blob))
        mean = arr[0]

    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    if args.gpu:
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    top_k = int(args.top_k)

    categories = []
    for line in np.loadtxt(args.synset_words, str, delimiter="\t"):
        print (line[1])
        categories.append(line[1])


    # Make classifier.
    print("load tranied model")
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, mean=mean.mean(1).mean(1),
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)

    # open csv file (for output)
    csvFileFP = open(args.output+"_top"+args.top_k+".csv", 'w')
    csvFile = csv.writer(csvFileFP,
                        delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvLine = [ 'file','ground truth' ]
    for i in range(top_k):
        csvLine.extend(['name','score'])
    csvFile.writerow(csvLine)

    # load file list
    g_start = time.time()
    lines = open(os.path.expanduser(args.test_file), 'r').readlines()
    for line_id, line in enumerate(lines):
        (filename, obj)=line[:-1].split(" ")
        inputs = [caffe.io.load_image(args.test_dir +"/" + filename) ];
        csvLine = [ filename, categories[int(obj)] ]
    
        # Classify.
        print("%d / %d Classifying %s." % (line_id+1, len(lines), filename))
        l_start = time.time()

        predictions = classifier.predict(inputs, not args.center_only)
        prediction = zip(predictions[0].tolist(), categories)
        prediction = sorted(prediction, key=lambda x: x[0], reverse=True)
        #prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
        print(" ground_truth: %s" % categories[int(obj)])
        for rank, (score, name) in enumerate(prediction[:top_k], start=1):
            print('  #%d | %s | %4.1f%%' % (rank, name, score * 100))
            csvLine.extend([name, str(score*100)])
        print(" Done in %.2f s." % (time.time() - l_start))
       
        csvFile.writerow(csvLine)
        csvFileFP.flush()

    print(" Globally Done in %.2f s." % (time.time() - g_start))

if __name__ == '__main__':
    main(sys.argv)
