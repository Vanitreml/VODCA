# -*- coding: utf-8 -*-
"""
Created on Sun May  4 21:24:14 2025

@author: vanni
"""

import os
import numpy as np
import csv
import re
import cv2 as cv
from PIL import Image
import Slider_VODCA_eng
import Auswertung_VODCA_eng
import glob


def work_through_folder(folder_directory, data_evaluation='yes', **kwargs):
    """
    unpacks all subfolders of the given directory
    Parameters
    ----------
    folder_directory : str
        path to the folder
    data_evaluation : str, optional
        should the data be evaluated? The default is 'yes'.
    **kwargs : TYPE
        optional arguments for Nm calculation (a,b,d)

    Returns
    -------
    None.

    """
    folders = os.listdir(folder_directory)
    folders = [f for f in os.listdir(folder_directory) if os.path.isdir(
        os.path.join(folder_directory, f))]

    for folder in folders:
        while True:  
            directory = os.path.join(folder_directory,folder)
            filename = os.path.basename(directory)
            print(f'currently folder {filename} is being evaluted')
            
            # creates a list with all the images in the directory
            all_images = glob.glob(os.path.join(directory, "*.jpg"))
            
            # calls the contourdetection function
            contours = recognize_contour(all_images[1], filename, folder_directory)
            print('contour detection finished, droplets are being counted...')
            contours = contours.tolist()
            
            #contour detection function is able to change the image path of image 1 so we have to extract it again
            all_images = glob.glob(os.path.join(directory, "*.jpg"))
            # creates a mask for the droplets, to be able to skip already as frozen detected droplets in further analysis
            contours_boolean = [[1] for i in contours]
            contours_YN = np.append(contours, contours_boolean, axis=1)

            prepare_csv(folder_directory, filename)
            array_ratio = []
            array_radius = []
    
            status, contours_YN, ratio, array_radius = main(all_images, filename, folder_directory, contours_YN, array_ratio, array_radius)
            
            if status == 'exit':
                
                return  
            elif status == 'retry':
                
                continue  
            elif status == 'go on':
                
                break 
    

def main(all_images, filename, folder_directory, contours_YN, array_ratio, array_radius):
    """
    this is the main function, calling all the important functions

    Parameters
    ----------
    directory : str
        path to the subfolder.
    folder_directory : str
        path to the main folder

    Returns
    -------
    None.

    """
    try:
        temperature = 0
        for i in range(len(all_images) - 1):

            # the temperature is extracted out of the name of the image
            if (float(cut_out_temperature(all_images[i + 1]))-temperature) < 0:
                break
            temperature = float(cut_out_temperature(all_images[i + 1]))

            # counts frozen droplets
            n_Tropfen, contours_YN, array_ratio, array_radius = count_frozen_droplets(
                all_images[i], all_images[i + 1], contours_YN, temperature, array_ratio, filename, folder_directory
            )

            # if there are any frozen droplets at a certain temperature, their parameters are saved in a csv file
            if n_Tropfen > 6:
                print('-------------------------------------------------')
                print('attention lots of frozen droplets are detected, check if lighting conditions were changed or sample was moved')
                print(f'current temperature: {temperature}')
                print('suggested workflow: check if something went wrong and if necessary delete the affected images')
                user_input = input("Do you want to continue ( ignore the issue) ? (go on/exit/retry): ")
                if user_input.lower() == "go on":
                    print("Continuing...")
                    print('-------------------------------------------------')
                    
                if user_input.lower() == 'retry':
                    print("Retrying")
                    print('-------------------------------------------------')
                    return 'retry', contours_YN, array_ratio, array_radius
                if user_input.lower() == 'exit':
                    print("Exiting...")
                    return 'exit', contours_YN, array_ratio, array_radius
            
                
            if n_Tropfen > 0:
                write_file(str(temperature), str(n_Tropfen), str(
                    array_radius), filename, folder_directory)
            

    except Exception as e:
        print(e)
    return 'go on', contours_YN, array_ratio, array_radius

def prepare_csv(folder_directory, filename):
    file_path = os.path.join(folder_directory, f'droplets_{filename}.csv')

    # deletes old evaluation files
    if os.path.exists(file_path):
        os.remove(file_path)

    # schreibt die Überschrift in die csv Datei
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['temperature', 'number of frozen droplets', 'radius frozen droplets'])
    
    
    
    
    
def image_names(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]


def recognize_contour(image, filename, folder_directory):
    """
    creates an object of the class InteractiveContourDetection
    """
    interactive = Slider_VODCA_eng.InteractiveContourDetection(
        image, filename, folder_directory)
    contours = interactive.show()
    return contours[0, :, :]


def write_file(string1, string2, string3, folder, directory):
    """
    writes the strings to a csv file

    """
    file_path = os.path.join(directory, f'droplets_{folder}.csv')
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([string1, string2, string3])


def cut_out_temperature(image):
    """
     searches for the temperature in the path of the image format: number + either ',' '.' + number

    Parameters
    ----------
    image : str
        path to image

    Returns
    -------
    temperature: float
        

    """
    temperature_match = re.search(r"\d+[\,\.]\d+", image)
    if temperature_match:
        temperature = temperature_match.group()

        # replaces ',' with '.'  as python uses '.' for decimals numbers
        return temperature.replace(',', '.') if ',' in temperature else temperature

    return None


def count_frozen_droplets(image1_file, image2_file, contours, temperature, array_ratio, filename, directory):
    """
    Counts the frozen droplets at a certain temperature
    
    Parameters
    ----------
    image1_file : str
        path to the first image
    image2_file : str
        path to the second image.
    contours : numpy array
        contains x, y coordinates and radius of the droplets
    temperature : float
        temperature.
    array_ratio : r
        DESCRIPTION.
    filename : str
        name of the subfolder.
    directory : str
        path to the folder.

    Returns
    -------
    n : int
        number of the droplet
    contours : numpy array
        contains x, y coordinates and radius of the droplets
    array_ratio : list
        containing the sum of differences/ Area
        .
    array_radius : np.array
        containing the radii

    """
    
    image1 = cv.cvtColor(np.array(Image.open(image1_file)), cv.COLOR_RGB2BGR)
    image2 = cv.cvtColor(np.array(Image.open(image2_file)), cv.COLOR_RGB2BGR)
    array_ratio.append(temperature)
    list_of_frozen_droplets_radii = [] # ähmmm????!!!!!!!!!!!!!!

    if contours is not None:
        contours = np.around(np.uint64(contours))
        n = 0
        for pt in contours[:]:
            
            # unpacks x,y,radius of contours
            x, y, r, YN = int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3])
            
            
            # cuts out the labelling of the picture in the right corner
            if r < 48 or (x + r > 1648 and y + r > 1445):
                YN = 0
                
            if YN:
                
               # cuts out the droplets of the two images and subtracts them from each other
                try:
                    height, width = image2.shape[:2]

                    x1 = max(x - r, 0)
                    x2 = min(x + r, width)
                    y1 = max(y - r, 0)
                    y2 = min(y + r, height)
                    
                    cropped_droplet2 = image2[y1:y2, x1:x2]
                    cropped_droplet1 = image1[y1:y2, x1:x2]
                    
                    cropped_subtracted_droplet = cv.subtract(cropped_droplet1, cropped_droplet2)
                    
                except Exception as e:
                    print(e, temperature)

        
                if cropped_subtracted_droplet is not None:
                    # sums up the differences between the matrix elements
                    sum_of_differences = abs(np.sum(cropped_subtracted_droplet))
                    
                    # the sum of differences is higher in bigger droplets, so we divide them by the area
                    ratio = sum_of_differences * 3.14 / ((r ) ** 2)
                    array_ratio.append(ratio)
                    
                    # if the sum of differences/Area exceeds a certain value they are detected as frozen.
                    if ratio > 50 and YN == True:
                        n += 1
                        #cv.circle(image2, (x, y), int(r) + 30, (50, 100, 182), 30)
                        list_of_frozen_droplets_radii.append(r)
                        pt[3] = 0

        array_radius = np.array(list_of_frozen_droplets_radii)/49*15
        array_radius = np.round(array_radius).astype(int)

    return n, contours, array_ratio, array_radius






