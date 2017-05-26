import scipy
import numpy as np
import skimage
from skimage.filters import gabor_kernel
from scipy.signal import convolve2d
from scipy import ndimage as ndi
from skimage import measure

from flika.roi import makeROI
from flika import global_vars as g
from flika.process.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox
from flika.process import difference_of_gaussians, threshold, zproject, remove_small_blobs
from flika.window import Window
from flika.process.file_ import open_file

def var_convoluted(I, N):
    im = np.array(I)
    im2 = im**2
    s = skimage.filters.gaussian(im, N)
    s2 = skimage.filters.gaussian(im2, N)
    return (s2 - s**2)

def reduce_local_contrast(I):
    I_var = var_convoluted(I, 6)
    I_var /= 1000
    I_var *= 50
    I_var += 1
    I = I / I_var
    return I

def generate_kernel(theta=0):
    """
    for testing
    ###########
    kernels = get_kernels()
    Window(kernels[0])
    axial_profile = np.mean(kernel, 1)
    pg.plot(axial_profile)
    
    parameters set 1
    ################
    frequency = .01
    sigma_x = 5
    sigma_y = 1
    #axial_profile = np.mean(kernel, 1)
    #kernel = kernel - axial_profile[:, np.newaxis]
    """
    # parameters set 2:
    frequency = .2
    sigma_x = .1  # left right axis. Bigger this number, smaller the width
    sigma_y = 2  # right left axis. Bigger this number, smaller the height
    kernel = np.real(gabor_kernel(frequency, theta, sigma_x, sigma_y))
    kernel -= np.mean(kernel)
    return kernel

def get_kernels():
    # prepare filter bank kernels
    kernels = []
    for theta in np.linspace(0, np.pi, 40):
        kernel = generate_kernel(theta)
        kernels.append(kernel)
    return kernels

def convolve_with_kernels(I, kernels):
    results = []
    for k, kernel in enumerate(kernels):
        print(k)
        filtered = ndi.convolve(I, kernel, mode='constant')
        results.append(filtered)
    results = np.array(results)
    return results

def convolve_with_kernels_fft(I, kernels):
    results = []
    for k, kernel in enumerate(kernels):
        print(k)
        filtered = scipy.signal.fftconvolve(I, kernel, 'same')
        results.append(filtered)
    results = np.array(results)
    return results

def get_lines(binary_window):
    lines = []
    for frame in np.arange(binary_window.mt):
        labelled = measure.label(binary_window.image[frame])
        for i in np.arange(np.max(labelled))+1:
            pos = np.argwhere(labelled == i)
            if pos.shape[1] == 3:
                pos = pos[:, 1:]
            leftpoint = pos[np.argmin(pos[:,0])]
            rightpoint = pos[np.argmax(pos[:, 0])]
            lines.append([leftpoint, rightpoint])
    return lines

def remove_overlapping_lines(lines):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # Return true if line segments AB and CD intersect
    def intersect(line1, line2):
        A, B = line1
        C, D = line2
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    overlapping = np.zeros((len(lines), len(lines)))
    for i in np.arange(len(lines)):
        for j in np.arange(i+1, len(lines)):
            if intersect(lines[i], lines[j]):
                overlapping[i, j] = 1
                overlapping[j, i] = 1
    lines_to_remove = []
    line_lengths = [np.sqrt(np.sum((l[1] - l[0]) ** 2)) for l in lines]
    overlap_pairs = np.argwhere(overlapping)
    for i in np.arange(len(lines)):
        partners = overlap_pairs[np.where(overlap_pairs[:, 0] == i)[0], 1]
        for partner in partners:
            if line_lengths[i] <= line_lengths[partner]:
                overlap_pairs[np.where(overlap_pairs[:, 0] == i)[0], :] = -1
                overlap_pairs[np.where(overlap_pairs[:, 1] == i)[0], :] = -1
                lines_to_remove.append(i)
                break
    lines_to_remove = np.array(lines_to_remove)
    new_lines = []
    for i in np.arange(len(lines)):
        if i not in lines_to_remove:
            new_lines.append(lines[i])
    return new_lines

kernels = get_kernels()


##### Test out the kernel #####
# kernel = generate_kernel(0)
# Window(kernel)


####################################
# STEP 1: Band Pass Spatial Filter
####################################
if __name__ == '__main__':
    original = open_file(r'Y:\Kyle\2017\2017_03_17_mda_cells_culture_TNT_detection_iansmith\trial_4.nd2')
    difference_of_gaussians(1.44, 2)
    zproject(0, g.currentWindow.mt, 'Max Intensity')
    Edges = difference_of_gaussians(.01, 2)
    threshold(0, keepSourceWindow=True)
    results = convolve_with_kernels_fft(g.currentWindow.image, kernels)
    Window(results) # zproject(0, g.currentWindow.mt, 'Max Intensity')
    binary_window = threshold(.1)
    binary_window = remove_small_blobs(2, 30)
    lines = get_lines(binary_window)
    #lines = remove_overlapping_lines(lines)
    Edges.setAsCurrentWindow()
    for line in lines:
        makeROI('line', line)




