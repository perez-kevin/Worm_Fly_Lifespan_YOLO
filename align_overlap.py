import cv2
import numpy as np

# align_image: use src1 as the reference image to transform src2
def align_image(src1, src2, warp_mode=cv2.MOTION_TRANSLATION):
    # convert images to grayscale
    img1_gray = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

    # define 2x3 or 3x3 matrices and initialize it to a identity matrix
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # number of iterations:
    num_iters = 1000

    # specify the threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-8    

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iters,  termination_eps)

    print('findTransformECC() may take a while...')

    # perform ECC: use the selected model to calculate the transformation required to align src2 with src1. The resulting transformation matrix is stored in warp_matrix:
    (cc, warp_matrix) = cv2.findTransformECC(img1_gray, img2_gray, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)

    if (warp_mode == cv2.MOTION_HOMOGRAPHY):
        img2_aligned = cv2.warpPerspective(src2, warp_matrix, (src1.shape[1], src1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # use warpAffine() for: translation, euclidean and affine models
        img2_aligned = cv2.warpAffine(src2, warp_matrix, (src1.shape[1], src1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return img2_aligned

# Align images

JpgDir = 'Images/Worms/'
Source = '1_7.jpg'
Target = '1_8.jpg'

imSource = cv2.imread(JpgDir + Source, cv2.IMREAD_COLOR)
imTarget = cv2.imread(JpgDir + Target, cv2.IMREAD_COLOR)

imReg2 = align_image(imSource, imTarget)

cv2.imwrite(JpgDir + Target + '_Aligned.jpg', imReg2)

# Overlap images

(B, G, R) = cv2.split(imSource)
(B1, G1, R1) = cv2.split(imReg2)
merged = cv2.merge([B, G, R1])

cv2.imwrite(JpgDir + Target + '_Overlapped.jpg', merged)