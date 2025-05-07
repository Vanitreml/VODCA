import numpy as np
import cv2 as cv
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os


class InteractiveContourDetection:
   
    
    def __init__(self, image_path, filename, folder_directory):
        self.image_path = image_path
        self.filename = filename
        self.folder_directory = folder_directory
        self.fig, self.ax = plt.subplots()
        self.sliders = {}
        self.ax_image = None
        self.current_contours = None
        self.window_closed = False



    def recognize_contour(self, param1, param2, minsize, maxsize):
        """
        evaluate the contours

        Parameters
        ----------
        param1 : float
            defines the sensitivity of the contour detection function.
        param2 : float
            defines the sensitivity of the contour detection function.
        minsize : float
            minimum size of the detected droplets
        maxsize : float
            maximum size of the detected droplets.

        Returns
        -------
   
        contours : numpy.ndarray
            x and y coordinates and the radius of the detected droplets

        """
        # for the contour detection function to work, we have to enhance the contrast and reduce noise
        # PIL Image.open can not work with str containing '°', so if the image path contains that sign it is replaced
        self.image_path = self.rename_first_image()
        image_1 = cv.imread(self.image_path)
        image_to_be_enhanced = Image.open(self.image_path)

        # blurs the image and enhances the contrast
        image_to_be_enhanced = image_to_be_enhanced.filter(ImageFilter.BLUR)
        enhancer = ImageEnhance.Contrast(image_to_be_enhanced)
        enhanced_image = enhancer.enhance(2)

        # cv2 and PIL use different data types, so now the PIL image is transformed
        gray_image1 = cv.cvtColor(np.array(enhanced_image), cv.COLOR_RGB2GRAY)

        # Gaussian blur is applied
        w_gray_image = cv.GaussianBlur(gray_image1, (21, 21), 0)
        blur = cv.GaussianBlur(w_gray_image, (31, 31), 0)

        # a bilateral Filter is applied to reduce unwanted noise  while keeping edges fairly sharp
        blur = cv.bilateralFilter(blur, 15, 150, 150)

        # the function assigns pixels the color black under a certain threshold and white above a certain thershold
        thresh_image = cv.adaptiveThreshold(
            blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 2)

        # the black-white image is filtered again
        thresh_image = cv.medianBlur(thresh_image, 9)

        # the contour detection function returns the x,y coordinates and radius of the droplets, the parameters 1 and 2 define the sensitivity
        contours = cv.HoughCircles(
            thresh_image, cv.HOUGH_GRADIENT, 1, 110, param1=param1, param2=param2, minRadius=minsize, maxRadius=maxsize)

        # The contours are drawn in the image
        if contours is not None:
            contours = np.around(np.uint16(contours))
            for pt in contours[0, :]:
                x, y, r = pt[0], pt[1], pt[2]
                if (x + r < 1648 or y + r < 1445):
                    cv.circle(image_1, (x, y), r + 30, (0, 0, 0), 13)

        return cv.cvtColor(image_1, cv.COLOR_BGR2RGB), contours

    def rename_first_image(self):
        unwanted_str = self.image_path
        newstr = unwanted_str.replace('\u00B0', "")
        os.rename(self.image_path, newstr)
        self.image_path = newstr
        return self.image_path

    def update(self, val):
        """
        passes the updated values of the sliders to the contour detection function


        """
        # links the sliders with the variables
        param1 = self.sliders['param1'].val
        param2 = self.sliders['param2'].val
        minsize = self.mu_to_pt(self.sliders['minsize'].val)
        maxsize = self.mu_to_pt(self.sliders['maxsize'].val)

        # executes the contour detection function with the new variables
        updated_image, contours = self.recognize_contour(
            param1, param2, minsize, maxsize)
        self.current_contours = contours
        self.ax_image.set_data(updated_image)
        self.fig.canvas.draw_idle()

    def on_close(self, event):
        self.window_closed = True

    def pt_to_mu(self, pt):
        # helps to calculate the radius in mu instead of pt
        return int(pt/49*15)

    def mu_to_pt(self, mu):
        # helps to calculate the radius in pt instead of mu
        return int(mu/15*49)

    def show(self):
        """
        displays the window for the interactive contour detection

        """
        

        # adjusts the distances between the sliders
        self.fig.subplots_adjust(left=0.1, bottom=0.3)

        # the contour detection function is called with some initial values
        result_image, contours = self.recognize_contour(12, 25, 49, 140)
        self.current_contours = contours

        # the image and the detected circles are displayed
        self.ax_image = self.ax.imshow(result_image, cmap='gray')
        self.ax.set_title("Interactive contour detection")
        # no ax labelling
        self.ax.axis('off')

        # definition of the sliders
        sliders_definitions = {
            'param1': [0.2, 0.2, "Param1", 10, 30, 12],
            'param2': [0.2, 0.15, "Param2", 15, 35, 25],
            'minsize': [0.2, 0.1, "Min Size ($\mu m$)", 0, self.pt_to_mu(300), self.pt_to_mu(49)],
            'maxsize': [0.2, 0.05, "Max Size ($\mu m$)", 0, self.pt_to_mu(300), self.pt_to_mu(140)],
        }

        # iterating through the slider definition dictionary
        for key, (x, y, label, min_val, max_val, init_val) in sliders_definitions.items():
            
            # creates and displays sliders
            ax_slider = plt.axes([x, y, 0.65, 0.03])
            slider = Slider(ax_slider, label, min_val, max_val,
                            valinit=init_val, valstep=1)

            # as soon as sliders are changed the update function is called
            slider.on_changed(self.update)

            # saves current sliders
            self.sliders[key] = slider

        # when the interactive plot is closed, the self.on_close function is called
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        plt.show()


        # window will be updated and displayed before the pause, and the GUI event loop will run during the pause
        while not self.window_closed:
            plt.pause(0.1)

        return self.current_contours
