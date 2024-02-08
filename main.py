import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from pathlib import Path

def find_template(img, template):
    w, h = template.shape[::-1]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    bboxes = []
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        # cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        bboxes.append((pt[0], pt[1], w, h))
    return bboxes

def draw_bboxes(img, bboxes):
    for box in bboxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
    return img

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    templates_files = list(map(str, Path("templates/").glob("*.jpg")))
    print(templates_files)
    templates = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in templates_files]
    # template = cv2.imread('me.jpg', cv2.IMREAD_GRAYSCALE)
    while(1):
        ret, img = cap.read()
    # img_rgb = cv2.imread('knot_detection.jpg')
    # assert img_rgb is not None, "file could not be read, check with os.path.exists()"
        all_boxes = []
        for tmp in templates:
            bboxes = find_template(img, tmp)
            all_boxes += bboxes
        draw_bboxes(img, all_boxes)
        cv2.imshow('res.jpg',img)
        # cv2.imwrite(f'templates/me_{time.time()}.jpg', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.5)

    cap.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 