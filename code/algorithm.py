# CSC320 Winter 2018
# Assignment 4
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np
# import the heapq package
from heapq import heappush, heappushpop, nlargest
# see below for a brief comment on the use of tiebreakers in python heaps
from itertools import count
import numpy as np
from heapq import heappush, heappop, heappushpop, nlargest
from copy import deepcopy as copy

_tiebreaker = count()

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the Generalized PatchMatch
# algorithm, as explained in Section 3.2 of the PatchMatch paper and Section 3
# of the Generalized PatchMatch paper.
#
# The function takes k NNFs as input, represented as a 2D array of heaps and an
# associated 2D array of dictionaries. It then performs propagation and random search
# as in the original PatchMatch algorithm, and returns an updated 2D array of heaps
# and dictionaries
#
# The function takes several input arguments:
#     - source_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the target image.
#     - f_heap:              For an NxM source image, this is an NxM array of heaps. See the
#                            helper functions below for detailed specs for this data structure.
#     - f_coord_dictionary:  For an NxM source image, this is an NxM array of dictionaries. See the
#                            helper functions below for detailed specs for this data structure.
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure
#     NOTE: the variables f_heap and f_coord_dictionary are modified in situ so they are not
#           explicitly returned as arguments to the function

def propagation_and_random_search_k(source_patches, target_patches,
                                    f_heap,
                                    f_coord_dictionary,
                                    alpha, w,
                                    propagation_enabled, random_enabled,
                                    odd_iteration,
                                    global_vars
                                    ):
    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    src_shp = source_patches.shape
    x, y = src_shp[0], src_shp[1]
    f, D = NNF_heap_to_NNF_matrix(f_heap)

    ## Find k (number of nnf displacements per point)
    k_size = len(f_heap[0][0])

    for i in range(x):
        for j in range(y):
            src_patch = source_patches[i, j, :, :]

            # start propagation
            if propagation_enabled:
                for k in range(k_size):
                    explored = [f[k][i][j]]
                    # Odd iterations
                    if odd_iteration:
                        # check if index is out of bound
                        if i != 0:
                            explored.append(f[k][i - 1][j])
                        if j != 0:
                            explored.append(f[k][i][j - 1])
                            # Even iterations
                    else:
                        # check if index is out of bound
                        if i < x - 1:
                            explored.append(f[k][i + 1][j])
                        if j < y - 1:
                            explored.append(f[k][i][j + 1])
                    explored = np.vstack(explored)
                    explored = explored.astype(int)
                    targets = explored + [i, j]
                    # loop over updated target patch
                    for tar in targets:
                        # check if target index is out of bound
                        tar_x, tar_y = tar[0], tar[1]
                        if 0 <= tar_x < x and 0 <= tar_y < y:
                            tar_patch = target_patches[tar_x, tar_y, :, :]
                            # get diff between source and patch
                            diff = (tar_patch - src_patch) ** 2
                            # use just non nan value to compute mean
                            non_nans = diff[~np.isnan(diff)]
                            cur_min = np.mean(non_nans)
                            cur_disp = [tar_x - i, tar_y - j]
                            cur_tuple = -cur_min, _tiebreaker.next(), cur_disp
                            if cur_min < D[k, i, j]:
                                key = tuple(cur_disp)
                                if key not in f_coord_dictionary[i][j]:
                                    # update values
                                    heappushpop(f_heap[i][j], cur_tuple)
                                    f_coord_dictionary[i][j][key] = 0
                                    f[k, i, j] = cur_disp
                                    D[k, i, j] = cur_min
            # start ramdom
            if random_enabled:
                max_iters = 1 + np.int(-np.log(w) / np.log(alpha))
                for k in range(k_size):
                    rand = np.random.uniform(-1, 1, (2, max_iters))
                    cur_f = f[k][i][j]

                    # offsets = []
                    # for i in range(1 + max_iters):
                    #     offsets.append(rand[:, i] * w * (alpha ** i))
                    def compute_rand(it):
                        return f[k][i][j] + rand[:, it] * w * (alpha ** it)

                    offsets = map(compute_rand, range(max_iters))
                    offsets = np.asarray(offsets).astype(int)
                    targets = offsets + [i, j]

                    # loop over updated target patch
                    for tar in targets:
                        tar_x, tar_y = tar[0], tar[1]
                        if 0 <= tar_x < x and 0 <= tar_y < y:
                            tar_patch = target_patches[tar_x, tar_y, :, :]
                            # get diff between source and patch
                            diff = (tar_patch - src_patch) ** 2
                            # use just non nan value to compute mean
                            non_nans = diff[~np.isnan(diff)]
                            cur_min = np.mean(non_nans)
                            cur_disp = [tar_x - i, tar_y - j]
                            cur_tuple = -cur_min, _tiebreaker.next(), cur_disp

                            # update value if smaller
                            if cur_min < D[k, i, j]:
                                key = tuple(cur_disp)
                                if key not in f_coord_dictionary[i][j]:
                                    heappushpop(f_heap[i][j], cur_tuple)
                                    f_coord_dictionary[i][j][key] = 0
                                    f[k, i, j] = cur_disp
                                    D[k, i, j] = cur_min

    #############################################

    return global_vars


# This function builds a 2D heap data structure to represent the k nearest-neighbour
# fields supplied as input to the function.
#
# The function takes three input arguments:
#     - source_patches:      The matrix holding the patches of the source image (see above)
#     - target_patches:      The matrix holding the patches of the target image (see above)
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds k NNFs. Specifically,
#                            f_k[i] is the i-th NNF and has dimension NxMx2 for an NxM image.
#                            There is NO requirement that f_k[i] corresponds to the i-th best NNF,
#                            i.e., f_k is simply assumed to be a matrix of vector fields.
#
# The function should return the following two data structures:
#     - f_heap:              A 2D array of heaps. For an NxM image, this array is represented as follows:
#                               * f_heap is a list of length N, one per image row
#                               * f_heap[i] is a list of length M, one per pixel in row i
#                               * f_heap[i][j] is the heap of pixel (i,j)
#                            The heap f_heap[i][j] should contain exactly k tuples, one for each
#                            of the 2D displacements f_k[0][i][j],...,f_k[k-1][i][j]
#
#                            Each tuple has the format: (priority, counter, displacement)
#                            where
#                                * priority is the value according to which the tuple will be ordered
#                                  in the heapq data structure
#                                * displacement is equal to one of the 2D vectors
#                                  f_k[0][i][j],...,f_k[k-1][i][j]
#                                * counter is a unique integer that is assigned to each tuple for
#                                  tie-breaking purposes (ie. in case there are two tuples with
#                                  identical priority in the heap)
#     - f_coord_dictionary:  A 2D array of dictionaries, represented as a list of lists of dictionaries.
#                            Specifically, f_coord_dictionary[i][j] should contain a dictionary
#                            entry for each displacement vector (x,y) contained in the heap f_heap[i][j]
#
# NOTE: This function should NOT check for duplicate entries or out-of-bounds vectors
# in the heap: it is assumed that the heap returned by this function contains EXACTLY k tuples
# per pixel, some of which MAY be duplicates or may point outside the image borders


def NNF_matrix_to_NNF_heap(source_patches, target_patches, f_k):
    f_heap = []
    f_coord_dictionary = []

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    f_shp = f_k.shape
    x, y = f_shp[1], f_shp[2]
    small = float('-inf')
    for i in range(x):
        # add row
        f_heap.append([])
        f_coord_dictionary.append([])
        for j in range(y):
            h = []
            d = {}
            dis_vec = f_k[:, i, j, :]
            src_patch = source_patches[i, j, :, :]
            for k in range(dis_vec.shape[0]):
                dx, dy = dis_vec[k, :]
                # if the index is invalid
                if i + dx >= x or j + dy >= y:
                    t = (small, _tiebreaker.next(), dis_vec[k, :])
                    heappush(h, t)
                else:
                    new_i, new_j = i + dx, j + dy
                    tar_patch = target_patches[new_i, new_j, :, :]
                    diff = (tar_patch - src_patch) ** 2
                    # just using non nan value to compute mean
                    not_nan = diff[~np.isnan(diff)]
                    sim = np.mean(not_nan)
                    t = (-sim, _tiebreaker.next(), dis_vec[k, :])
                    heappush(h, t)
                key = tuple([dx, dy])
                d[key] = 0
            f_heap[i].append(h)
            f_coord_dictionary[i].append(d)

    #############################################

    return f_heap, f_coord_dictionary


# Given a 2D array of heaps given as input, this function creates a kxNxMx2
# matrix of nearest-neighbour fields
#
# The function takes only one input argument:
#     - f_heap:              A 2D array of heaps as described above. It is assumed that
#                            the heap of every pixel has exactly k elements.
# and has two return arguments
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds the k NNFs represented by the heap.
#                            Specifically, f_k[i] should be the NNF that contains the i-th best
#                            displacement vector for all pixels. Ie. f_k[0] is the best NNF,
#                            f_k[1] is the 2nd-best NNF, f_k[2] is the 3rd-best, etc.
#     - D_k:                 A numpy array of dimensions kxNxM whose element D_k[i][r][c] is the patch distance
#                            corresponding to the displacement f_k[i][r][c]
#

def NNF_heap_to_NNF_matrix(f_heap):
    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    k = len(f_heap[0][0])
    shp = [len(f_heap), len(f_heap[0])]
    x, y, k = len(f_heap), len(f_heap[0]), len(f_heap[0][0])

    D_k = np.zeros((k, x, y))
    f_k = np.zeros((k, x, y, 2))

    for i in range(x):
        for j in range(y):
            # find largest elements list and update value
            l = nlargest(k, f_heap[i][j])
            for ind in range(len(l)):
                # update values
                f_k[ind, i, j] = l[ind][2]
                D_k[ind, i, j] = -l[ind][0]
    f_k = f_k.astype(int)

    #############################################

    return f_k, D_k


def nlm(target, f_heap, h):
    # this is a dummy statement to return the image given as input
    # denoised = target

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    #
    t_shp = target.shape
    x, y, k = t_shp[0], t_shp[1], len(f_heap[0][0])
    denoised = np.zeros(t_shp)
    # get target matrix
    t_matrix = make_coordinates_matrix(t_shp)
    f_k, D_k = NNF_heap_to_NNF_matrix(f_heap)
    tar_index = (t_matrix + f_k)
    tar_index.shape = (-1, 2)
    tar_x, tar_y = tar_index[:, 0], tar_index[:, 1]
    targets = target[tar_x, tar_y]
    targets.shape = (-1, x, y, 3)
    # compute w value use formula in section 3 in paper
    e_value = np.exp(-(D_k ** 0.5 / h ** 2))
    Z = np.sum(e_value, axis=0)
    w = e_value / Z
    # update denoised value
    for i in range(x):
        for j in range(y):
            for ki in range(k):
                denoised[i, j] += targets[ki, i, j] * w[ki, i, j]

    #############################################

    return denoised


#############################################
###  PLACE ADDITIONAL HELPER ROUTINES, IF ###
###  ANY, BETWEEN THESE LINES             ###
#############################################


#############################################


# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    rec_source = None

    ################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES  ###
    ################################################
    t_shp, f_shp = target.shape, f.shape
    t = make_coordinates_matrix(t_shp) + f
    t.shape = (-1, 2)
    x, y = t[:, 0], t[:, 1]
    rec_source = target[x, y]
    rec_source.shape = (-1, f_shp[1], 3)

    #############################################

    return rec_source


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
