from argparse import ArgumentParser
from utils import *
from images import *
from feature import *
from match import *


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

    img1 = images[4]
    img2 = images[3]
    keyPoints1 = harris_detector(img1)
    desc1 = keypoint_descriptor(img1, keyPoints1)

    keyPoints2 = harris_detector(img2)
    desc2 = keypoint_descriptor(img2, keyPoints2)

    matches = find_matches(desc1, desc2, 0.8)
    # print(len(matches))
    # show_match(img1, img2, matches)
    bestdyx = ransac(matches)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)