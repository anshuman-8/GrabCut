import cv2
import numpy as np
from collections import namedtuple

con = namedtuple('_', ('FIX', 'UNK', 'FG', 'BG'))(1, 0, 1, 0)

class EventHandler:
    """
    Class for handling user input during segmentation iterations 
    """
    
    def __init__(self, flags, img, _types, _alphas, colors):
        
        self.FLAGS = flags
        self.ix = -1
        self.iy = -1
        self.img = img
        self.img2 = self.img.copy()
        self._types = _types
        self._alphas = _alphas
        self.COLORS = colors

    @property
    def image(self):
        return self.img
    
    @image.setter
    def image(self, img):
        self.img = img
        
    @property
    def types(self):
        return self._types

    @types.setter
    def types(self, _types):
        self._types = _types
    
    @property
    def alphas(self):
        return self._alphas

    @alphas.setter
    def alphas(self, _alphas):
        self._alphas = _alphas
    
    @property
    def flags(self):
        return self.FLAGS 
    
    @flags.setter
    def flags(self, flags):
        self.FLAGS = flags
    
    def handler(self, event, x, y, flags, param):

        # Draw the rectangle first
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.FLAGS['rect_over'] == False:
                self.FLAGS['DRAW_RECT'] = True
                self.ix, self.iy = x,y

        if event == cv2.EVENT_MOUSEMOVE:
            if self.FLAGS['DRAW_RECT'] == True:
                self.img = self.img2.copy()
                cv2.rectangle(self.img, (self.ix, self.iy), (x, y), self.COLORS['BLUE'], 2)
                self.FLAGS['RECT'] = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
                self.FLAGS['rect_or_mask'] = 0

        if event == cv2.EVENT_LBUTTONUP:
            if self.FLAGS['rect_over'] == False:
                self.FLAGS['DRAW_RECT'] = False
                self.FLAGS['rect_over'] = True
                cv2.rectangle(self.img, (self.ix, self.iy), (x, y), self.COLORS['BLUE'], 2)
                self.FLAGS['RECT'] = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
                self.FLAGS['rect_or_mask'] = 0

                # initialise types and alphas
                temp = np.zeros(self._types.shape, dtype=np.uint8)
                rect = self.FLAGS['RECT']
                temp[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = 1
                self._types[temp == 0] = con.FIX
                self._alphas[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = con.FG

        
        # Draw strokes for refinement 
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.FLAGS['rect_over']:
                self.FLAGS['DRAW_STROKE'] = True
                self.prev_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.FLAGS['DRAW_STROKE']:
                cv2.line(self.img, self.prev_point, (x, y), self.FLAGS['value']['color'], 3)
                cv2.line(self._alphas, self.prev_point, (x, y), self.FLAGS['value']['val'], 3)
                cv2.line(self._types, self.prev_point, (x, y), con.FIX, 3)
                self.prev_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.FLAGS['DRAW_STROKE']:
                self.FLAGS['DRAW_STROKE'] = False
                cv2.line(self.img, self.prev_point, (x, y), self.FLAGS['value']['color'], 3)
                cv2.line(self._alphas, self.prev_point, (x, y), self.FLAGS['value']['val'], 3)
                cv2.line(self._types, self.prev_point, (x, y), con.FIX, 3)