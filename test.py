import os
import cv2
import numpy as np
from PIL import Image
import json


img_path = os.path.join(self.path, self.filenames[idx])
        img = np.asarray_chkfinite(Image.open(img_path))
        with open(os.path.join(self.json_path, self.filenames[idx]), 'r') as f:
            json_data = json.load(f)
        
        for i in range(0,len(json_data['annotations'])):
            point = np.array(json_data['annotations'][i]['points'])
            label_image = cv2.polylines(img, [point], True)
            label_image = cv2.fillPoly(label_image, [point], (200, 255, 200))