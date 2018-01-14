import winsound

import cv2
import numpy as np

learning_parameter = 0.005


def main():
    cam = cv2.VideoCapture(0)
    sub_mog2 = cv2.createBackgroundSubtractorMOG2()
    while True:
        ret_val, img = cam.read()
        img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (0, 0), fx=0.25, fy=0.25)
        mask_sub_mog2 = sub_mog2.apply(img_gray, learningRate=learning_parameter)
        mask_binary = np.array(mask_sub_mog2 >= 127, dtype='uint8')
        move_percentage = np.sum(mask_binary) / mask_sub_mog2.size
        frequency = int(3800 * move_percentage / 2 + 100)
        winsound.Beep(frequency, 100)

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
