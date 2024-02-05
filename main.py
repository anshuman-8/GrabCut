import cv2

from src.run import run

if __name__ == '__main__':
    filename = './images/ceramic.jpg'
    try:
        run(filename)
    finally:
        cv2.destroyAllWindows()