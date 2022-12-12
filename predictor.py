import numpy as np
import cv2

import time

IMG_PER_FILE = 30

IMG_SIZE = 128

CHANNELS = 3


def video_reader(cv2, filename):
    frames = np.zeros((IMG_PER_FILE, IMG_SIZE, IMG_SIZE,
                      CHANNELS), dtype=np.float16)
    i = 0
    # print(frames.shape)
    vc = cv2.VideoCapture(filename)

    #print("reading video")
    while i < IMG_PER_FILE:
        success, frame = vc.read()
        # if success:
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frm = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))

        frm = np.expand_dims(frm, axis=0)
        # rescale [0,1]
        frm = frm / 255.0
        frames[i][:] = frm
        i += 1
    datav = np.zeros((1, 30, IMG_SIZE, IMG_SIZE, 3), dtype=np.float16)
    datav[0][:][:] = frames
    
    vc.release()
    return datav


def pred_fight(model, video, acuracy=0.9):
    pred_test = model.predict(video)
    if pred_test[0][1] >= acuracy:
        return True, pred_test[0][1]
    else:
        return False, pred_test[0][1]


def main_fight(video, model):
    vid = video_reader(cv2, video)
    millis = int(round(time.time() * 1000))
    # print(millis)
    f, precent = pred_fight(model, vid, acuracy=0.65)
    millis2 = int(round(time.time() * 1000))
    # print(millis2)
    res_fight = {'violence': f, 'violence estimation': str(precent)}
    res_fight['processing_time'] = str(millis2-millis)
    return res_fight
