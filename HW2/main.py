from argparse import ArgumentParser
from utils import *
from images import *
from feature import *
from match import *
from projection import *
from Alignment import *


parser = ArgumentParser('High Dynamic Range Imaging')
parser.add_argument('--dataset', default='parrington', help='Name of input dataset.')
parser.add_argument('--ratio', default='1', type=int, help='Reshape ratio.')
parser.add_argument('--right', dest='left', action='store_false', help='Start from right.')
parser.add_argument('--left', dest='left', action='store_true', help='Start from left.')
parser.add_argument('--align', dest='align', action='store_true')
parser.add_argument('--no-align', dest='align', action='store_false')
parser.set_defaults(left=False)
parser.set_defaults(align=False)


def main(args):
    dataset = args.dataset
    reshapeRatio = args.ratio

    inputPath = 'data/'+dataset
    images, focals = read_image(inputPath)
    images = reshape_images(images, reshapeRatio)
    images = cylindrical_projection(images, focals)

    
    if args.left:
        order = range(0, len(images)-1, 1)
    else:
        order = range(len(images)-1, 0, -1)
    
    result = images[order[0]]
    shift = []
    for i in order:
        print(i)
        if args.left:
            img1 = images[i]
            img2 = images[i+1]
        else:
            img1 = images[i]
            img2 = images[i-1]
        keyPoints1 = harris_detector(img1)
        desc1 = keypoint_descriptor(img1, keyPoints1)


        keyPoints2 = harris_detector(img2)
        desc2 = keypoint_descriptor(img2, keyPoints2)

        matches = find_matches(desc1, desc2, 0.8)
        bestdyx = ransac(matches, 1000, 3)
        shift.append(bestdyx)
        result = combine_matches(result, img2, bestdyx)
        # show_image(result)
    resultPath = 'result/'+dataset+'_proj.png'
    cv2.imwrite(resultPath, result)
    # show_image(result)
    
    if(args.align):
        keyPoints1 = harris_detector(images[order[0]])
        desc1 = keypoint_descriptor(images[order[0]], keyPoints1)
        keyPoints2 = harris_detector(images[order[-1]])
        desc2 = keypoint_descriptor(images[order[-1]], keyPoints2)
        matches = find_matches(desc1, desc2, 0.8)
        bestdyx = ransac(matches, 1000, 3)
        
        resultPath = 'result/'+dataset+'_align.png'
        result_align = End2endAlignment(result,bestdyx)    
        cv2.imwrite(resultPath,result_align)
        # show_image(result_align)
    else:
        result_align = result

    resultPath = 'result/'+dataset+'_crop.png'
    result_crop = crop(result_align)    
    cv2.imwrite(resultPath,result_crop)
    show_image(result_crop)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)