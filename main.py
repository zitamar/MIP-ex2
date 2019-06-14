from copy import deepcopy

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage import morphology
import scipy
import time

def SegmentationByTH(niftyFile, Imin, Imax):
    """
    save the segmentation
    :param niftyFile:
    :param Imin:
    :param Imax:
    :return:
    """
    img = nib.load(niftyFile)
    name = img.get_filename()
    print(name)
    # print ("loaded image type " , img.get_data_dtype())
    img_data = img.get_data()
    low_values_flags = img_data < Imin
    high_values_flags = img_data > Imax
    img_data[:] = 1
    img_data[low_values_flags] = 0
    img_data[high_values_flags] = 0
    im_string = niftyFile.split('.')[0]
    # name_format = img.get_filename() + '__seg__' + str(Imin) + '_' + str(Imax) + '.nii.gz'
    name_format = im_string + '__seg__' + str(Imin) + '_' + str(Imax) + '.nii.gz'
    nib.save(img, name_format)
    # nib.save(img,"afterth")
    return img_data # saves the segmentation



def SegmentationTHgetsImg_Data (img_data, Imin, Imax):
    """
    problems occur when i sent the img when i need to make iterations, so this functions just gets the img_data
    and makes thresholding
    :param img_data:
    :param Imin:
    :param Imax:
    :return:
    """
    low_values_flags = img_data < Imin
    high_values_flags = img_data > Imax
    img_data[:] = 1
    img_data[low_values_flags] = 0
    img_data[high_values_flags] = 0
    # nib.save(img,"afterth")
    return 0 # saves the segmentation



def find_local_maximas (histogram):
    """
    find all local maximas along array
    :param histogram:
    :return:
    """
    maximas = []

    for i in range(205):
        j = i + 25
        k = i + 50
        if ((histogram[j]-histogram[i])>25) & ((histogram[j]-histogram[k])>25):
            maximas.append(j)


    return maximas





def find_second_peak(histogram):

    """
    it seems that the aorta greyscales always appear in the second peak of the hostogram. Therefore, i'd like to
    use this peak in order to find the most important grayscales.
    in the peak maximum is not in between 120-140, will return 130.
    :param maximas:
    :return:
    """

    maximas = find_local_maximas(histogram)

    first_peak = []
    second_peak = []

    first_or_second = 1

    print("maximas are " , maximas)

    for i in range(len(maximas)-1):
        j = i + 1

        if first_or_second >2:
            break

        if (maximas[j]-maximas[i])>30:
            first_or_second+=1
        if (first_or_second==1):
            first_peak.append(maximas[j])
        if(first_or_second==2):
            second_peak.append(maximas[j])


    return second_peak




def find_second_peak_highest_value(histogram):
    """
    finds the highest
    :param histogram:
    :return:
    """
    second_peak = find_second_peak(histogram)
    print("second is", second_peak)
    if (len(second_peak)) == 0:
        return 80

    maximum_gray_peak = second_peak[0]

    for i in second_peak:
        print("i and hist are", i, histogram[i], "\n")
        if (histogram[i]) > histogram[maximum_gray_peak]:

            maximum_gray_peak = i
            print ("max is", maximum_gray_peak)


    if (maximum_gray_peak > 140) | (maximum_gray_peak < 110):

        return 120


    print("return gray peak" , maximum_gray_peak)




    return maximum_gray_peak




def SkeletonTHFinder(niftyFile):
    """
    finds skeleton of given ct
    :param niftyFile:
    :return:
    """

    img = nib.load(niftyFile)
    img_data = img.get_data()
    # SegmentationByTH(img)
    SegmentationTHgetsImg_Data(img_data,150,1300)

    label , connectivity = measure.label(img_data,return_num=True)
    print("first_connectifity", connectivity)

    connectivities = []
    imin = []
    min_connectivity = connectivity
    imin_connectivity = 150
    final_connecticity = 150
    index_local_minima = 0
    for i in range(164, 500, 14):
        print("i is" , i)
        # nib.save(img, 'in first iteration.nii.gz')

        img_data=SegmentationByTH(niftyFile,i,1300)

        label , connectivity = measure.label(img_data,return_num=True)
        print("connectivity" , connectivity)
        connectivities.append(connectivity)
        imin.append(i)
        if connectivity < min_connectivity:
            if index_local_minima == 0:
                min_connectivity = connectivity # now it takes global minima
                imin_connectivity = i
        if connectivity>min_connectivity:
            print("found minima is" , imin_connectivity)
            index_local_minima +=1
    name_format = niftyFile + '_Graph.jpg'
    plt.plot(imin, connectivities)
    plt.grid()
    plt.xlabel('Minimal Grayscale')
    plt.ylabel('connected components')
    plt.title('connected component as function of minimal grayscale value')
    plt.savefig(name_format)
    plt.show()

    i = 1

    im_string = niftyFile.split('.')[0]

    string_img_to_load = im_string + '__seg__' + str(imin_connectivity) + '_' + str(1300) + '.nii.gz'


    oneConnectorComponent(string_img_to_load)



def oneConnectorComponent(segmentation,aorta=0, after = 0):
    """
    opens a segmentation and make 1 connector component, as well as other morphological operations :
    erosion and dialition
    :param segmentation: already been saved
    :param aorta:
    :param after:
    :return:
    """
    time.sleep(5) # solved some bugs.
    print("we open" , segmentation)
    seg = nib.load(segmentation)
    seg_data = seg.get_data()
    nib.save(seg,"afterOpenAndCloseSegmentation.nii.gz") # todo severe bug

    label, connectivity = measure.label(seg_data,return_num=True)
    # morphology.dilation(label)

    first_removal = 20000
    additional_removal = 1000


    if (aorta ==1):
        first_removal = 300
        additional_removal = 200
    removal = first_removal

    # label = morphology.erosion(label)

    while (connectivity >1):
        removal += additional_removal
        label = morphology.remove_small_objects(label,removal)
        label, connectivity = measure.label(label,return_num=True)
        print(connectivity)
        print(removal)
    #   morphology.remove_small_holes(label)


    if (aorta==1):
        if (after == 0): #after means if we have already try this before and got 0 connector components
            label = morphology.binary_erosion(label)


    label, connectivity = measure.label(label,return_num=True)
    print("connectivity after 1 erosions" , connectivity)


    while (connectivity >1):
        removal += additional_removal
        label = morphology.remove_small_objects(label,removal)
        label, connectivity = measure.label(label,return_num=True)
        print(connectivity)
        print(removal)

    if(connectivity==0): # make sure we have at least 1 connector component
        oneConnectorComponent(segmentation,1,1)


    label = morphology.dilation(label)
    label = morphology.binary_dilation(label)

    low_values_flags = label == False
    high_values_flags = label ==True
    seg_data[high_values_flags] = 1
    seg_data[low_values_flags] = 0
    im_string = segmentation.split('_')[0] +"SkeletonSegmentation.nii.gz"
    if aorta==1:
        im_string = segmentation.split('_')[0] + "aortaSegmentation.nii.gz"
    nib.save(seg,im_string)



def find_box_of_segmentation(segmentationPath):

    segmentation =  nib.load(segmentationPath)
    seg_data = segmentation.get_data()
    print(seg_data)
    indexes_in_x = np.where(segmentation == 1)

    print (indexes_in_x)


def create_histogram_to_cube(img_data):
    hist_orig, bins = np.histogram(img_data, 256, [0, 256])
    return hist_orig


def AortaSegmentatin (niftyFile, L1_seg):
    img = nib.load(niftyFile)
    img_data = img.get_data()
    # low_values_flag = img_data < 110 # save in advance the wanted values to zeros
    # high_values_flags = img_data > 200 # save in advance the wanted values to zeros
    seg = nib.load(L1_seg)
    seg_data = seg.get_data()
    # find ROI
    tuple_of_nonzeros= np.nonzero(seg_data)
    x_axis = tuple_of_nonzeros[0]
    y_axis = tuple_of_nonzeros[1]
    z_axis = tuple_of_nonzeros[2]
    minimal_z = np.amin(z_axis)
    maximal_z = np.amax(z_axis)
    minimal_x = np.amin(x_axis)
    maximal_x=np.amax(x_axis)
    minimal_y=np.min(y_axis)
    maximal_y=np.max(y_axis)
    x_min_index = np.floor((minimal_x+maximal_x)/2)
    x_min_index=x_min_index.astype(int)
    x = img_data[minimal_x:x_min_index,(minimal_y-(minimal_y/6)):(minimal_y-1),minimal_z:maximal_z]
    # x = img_data[minimal_x:x_min_index,minimal_y:maximal_y,minimal_z:maximal_z]

    cube_hist = create_histogram_to_cube(x)
    second_peak_hightest_value = find_second_peak_highest_value(cube_hist)
    low_values_flag = img_data < (second_peak_hightest_value-20) # save in advance the wanted values to zeros
    high_values_flags = img_data > second_peak_hightest_value+60 # save in advance the wanted values to zeros
    seg_data[:]=0 # everything is 0
    # seg_data[minimal_x:x_min_index,(minimal_y-(minimal_y/5)):(minimal_y-1),minimal_z:maximal_z] = 1 #
    seg_data[minimal_x:x_min_index,minimal_y:maximal_y,minimal_z:maximal_z] = 1 #

    nib.save(seg,'colorize the seg axis_change_y.nii.gz')
    img_data[:]=0 # # everything is 0
    img_data[minimal_x:x_min_index,(minimal_y-(minimal_y/6)):(minimal_y-1),minimal_z:maximal_z] = 1 # inside cube is
    img_data[low_values_flag] = 0
    img_data[high_values_flags]=0
    name_format = niftyFile.split('.')[0] + "__rectangle.nii.gz"
    nib.save(img,name_format)

    oneConnectorComponent(name_format,1)

def evaluateSegmentation (groundTruthSeg, estimatedSeg):
    groundTruth = nib.load(groundTruthSeg)
    groundTruthData = groundTruth.get_data()
    seg = nib.load(estimatedSeg)
    seg_data = seg.get_data()
    # find ROI
    tuple_of_nonzeros= np.nonzero(seg_data)
    x_axis = tuple_of_nonzeros[0]
    y_axis = tuple_of_nonzeros[1]
    z_axis = tuple_of_nonzeros[2]
    minimal_z = np.amin(z_axis)
    maximal_z = np.amax(z_axis)
    minimal_x = np.amin(x_axis)
    maximal_x=np.amax(x_axis)
    minimal_y=np.min(y_axis)
    maximal_y=np.max(y_axis)
    low_values_flag = groundTruthData <1

    groundTruthData[:] = 0
    groundTruthData[minimal_x:maximal_x,minimal_y:maximal_y,minimal_z:maximal_z] = 1
    nib.save(groundTruth,"cube.nii.gz")

    groundTruthData[low_values_flag] = 0

    nib.save(groundTruth,"sliceGT.nii.gz")


    union = np.logical_or(seg_data,groundTruthData)
    union = np.sum(union)
    intersection = np.logical_and(seg_data,groundTruthData)
    intersection = np.sum(intersection)


    divis = np.float(np.float(intersection) / np.float(union))


    #
    DC = 2*(divis)
    VOD = (1-divis)



    print ("d and v " , DC, VOD)



def FindMaxima(numbers):
    maxima = []
    length = len(numbers)
    if length >= 2:
        if numbers[0] > numbers[1]:
            maxima.append(numbers[0])

        if length > 3:
            for i in range(1, length-1):
                if numbers[i] > numbers[i-1] and numbers[i] > numbers[i+1]:
                    maxima.append(numbers[i])

        if numbers[length-1] > numbers[length-2]:
            maxima.append(numbers[length-1])
    return maxima


if __name__ == '__main__':


    print("eval is" , evaluateSegmentation("Case4_Aorta.nii.gz", "Case4aortaSegmentation.nii.gz"))