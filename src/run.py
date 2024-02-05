import cv2
import numpy as np
from collections import namedtuple

from src.event_handler import EventHandler
from src.grabcut import fit_gmms, grab_cut


con = namedtuple('_', ('FIX', 'UNK', 'FG', 'BG'))(1, 0, 1, 0)
NUM_GMM_COMP = 5
GAMMA = 50
LAMDA = 9 * GAMMA
NUM_ITERS = 10
TOL = 1e-3

def run(filename, n_components=NUM_GMM_COMP, gamma=GAMMA, lamda=LAMDA,
        num_iters=NUM_ITERS, tol=TOL, connect_diag=True):
    """
    Main loop that implements GrabCut. 
    
    Input
    -----
    filename (str) : Path to image
    """
    
    COLORS = {
    'BLACK' : [0,0,0],
    'RED'   : [0, 0, 255],
    'GREEN' : [0, 255, 0],
    'BLUE'  : [255, 0, 0],
    'WHITE' : [255,255,255]
    }

    DRAW_BG = {'color' : COLORS['BLACK'], 'val' : con.BG}
    DRAW_FG = {'color' : COLORS['WHITE'], 'val' : con.FG}

    FLAGS = {
        'RECT' : (0, 0, 1, 1),
        'DRAW_STROKE': False,         # flag for drawing strokes
        'DRAW_RECT' : False,          # flag for drawing rectangle
        'rect_over' : False,          # flag to check if rectangle is  drawn
        'rect_or_mask' : -1,          # flag for selecting rectangle or stroke mode
        'value' : DRAW_FG,            # drawing strokes initialized to mark foreground
    }

    img = cv2.imread(filename)
    img2 = img.copy()
    types = np.zeros(img.shape[:2], dtype = np.uint8)  # whether a pixel is known or unknown
    alphas = np.zeros(img.shape[:2], dtype = np.uint8) # mask is a binary array with : 0 - background pixels
                                                       #                               1 - foreground pixels 
    output = np.zeros(img.shape, np.uint8)             # output image to be shown

    # Input and segmentation windows
    cv2.namedWindow('Input Image')
    cv2.namedWindow('Segmented output')
    
    EventObj = EventHandler(FLAGS, img, types, alphas, COLORS)
    cv2.setMouseCallback('Input Image', EventObj.handler)
    cv2.moveWindow('Input Image', img.shape[1] + 10, 90)

    while(1):
        
        img = EventObj.image
        types = EventObj.types
        alphas = EventObj.alphas
        FLAGS = EventObj.flags
        cv2.imshow('Segmented image', output)
        cv2.imshow('Input Image', img)
        
        k = cv2.waitKey(1)

        # key bindings
        if k == 27:
            # esc to exit
            break
        
        elif k == ord('0'): 
            # Strokes for background
            FLAGS['value'] = DRAW_BG
        
        elif k == ord('1'):
            # FG drawing
            FLAGS['value'] = DRAW_FG
        
        elif k == ord('r'):
            # reset everything
            FLAGS['RECT'] = (0, 0, 1, 1)
            FLAGS['DRAW_STROKE'] = False
            FLAGS['DRAW_RECT'] = False
            FLAGS['rect_or_mask'] = -1
            FLAGS['rect_over'] = False
            FLAGS['value'] = DRAW_FG
            img = img2.copy()
            types = np.zeros(img.shape[:2], dtype = np.uint8) 
            alphas = np.zeros(img.shape[:2], dtype = np.uint8)
            EventObj.image = img
            EventObj.types = types
            EventObj.alphas = alphas
            output = np.zeros(img.shape, np.uint8)
        
        elif k == 13: 
            # Press carriage return to initiate segmentation
            
            #-------------------------------------------------#
            # Implement GrabCut here.                         #  
            # Function should return a mask which can be used #
            # to segment the original image as shown on L90   # 
            #-------------------------------------------------#
            alphas = grab_cut(img2, types, alphas, n_components, gamma, lamda, num_iters, tol, connect_diag)
            EventObj.alphas = alphas

        
        EventObj.flags = FLAGS
        mask2 = np.where((alphas == 1), 255, 0).astype('uint8')
        output = cv2.bitwise_and(img2, img2, mask = mask2)