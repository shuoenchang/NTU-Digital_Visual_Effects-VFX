from argparse import ArgumentParser
from utils import *
from images import *
from feature import *


parser = ArgumentParser('High Dynamic Range Imaging')
parser.add_argument('--dataset', default='parrington', help='Name of input dataset.')
parser.add_argument('--ratio', default='1', type=int, help='Reshape ratio.')
# parser.add_argument('--align', dest='align', action='store_true')
# parser.add_argument('--no-align', dest='align', action='store_false')
# parser.set_defaults(align=True)


def main(args):
    dataset = args.dataset
    reshapeRatio = args.ratio

    inputPath = 'data/'+dataset
    images, exposureTimes = read_image(inputPath)
    images = reshape_images(images, reshapeRatio)
    keyPoints = harris_detector(images[0])
    show_feature(images[0], keyPoints)
    keypoint_descriptor(images[0], keyPoints)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)