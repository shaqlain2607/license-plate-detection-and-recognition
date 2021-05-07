import cv2
import numpy as np
import scipy.ndimage
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer as lb
import pandas as pd
from skimage.segmentation import clear_border

def filter(img):
    img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT)
    
    # contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # print(len(contours))
    # contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    # meanArea = 0
    # for cnt in contours:
    #     meanArea = meanArea+cv2.contourArea(cnt)
    # meanArea = meanArea/len(contours)
    # print(meanArea)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, None, None, None, 8, cv2.CV_32S)
    result = np.zeros((img.shape), np.uint8)
    sizes= stats[:,-1]
    max_label= 1
    max_size= sizes[1]
    
    # nlabels= sorted(nlabels, key=lambda x: stats[x, cv2.CC_STAT_AREA], reverse=True)
    # print(nlabels)
    # result[labels==1]=255
    for i in range(2, nlabels):
         if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
            
    result[labels==max_label]=255          
    result = cv2.resize(result, (28, 28), cv2.INTER_AREA)

    return result


def extract_character(image):
    if(image.shape[0]>image.shape[1]):
        image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('img', image)
    cv2.waitKey(0)
    image = cv2.resize(image, (min(320, image.shape[1]*2), min(160, image.shape[0]*2)))
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # gray = cv2.resize(gray, (gray.shape[1]*2,gray.shape[0]*2))
    dim = gray.shape
    # print(dim)
    thresh = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # thresh=cv2.GaussianBlur(thresh, (3,3), 0)
    thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 1)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # thresh = cv2.adaptiveThreshold(thresh,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2)

    # kernel = np.ones((2,2), dtype= np.uint8)
    # thresh= cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # thresh = cv2.dilate(thresh, kernel, iterations = 1)
    cv2.imshow('thresh', thresh)
    cv2.imwrite('thresh.jpg', thresh)
    cv2.waitKey(0)
    thresh = clear_border(thresh)
    # cv2.imshow('thresh1', thresh)
    # cv2.waitKey(0)
    
    thresh = scipy.ndimage.median_filter(thresh, (5, 1))
    # thresh = scipy.ndimage.median_filter(thresh, (5, 1))
    # thresh = scipy.ndimage.median_filter(thresh, (1, 5))
    # thresh= cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imshow('thresh2', thresh)
    cv2.imwrite('thresh2.jpg', thresh)
    cv2.waitKey(0)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, None, None, 8, cv2.CV_32S)
    result = np.zeros(gray.shape, dtype="uint8")
    sum_w = 0
    sum_h = 0
    sum_area = 0

    for i in range(1, nlabels):
        sum_w = sum_w+stats[i, cv2.CC_STAT_WIDTH]
        sum_h = sum_h+stats[i, cv2.CC_STAT_HEIGHT]
        sum_area = sum_area+stats[i, cv2.CC_STAT_AREA]
    sum_w = sum_w/(nlabels-1)
    sum_h = sum_h/(nlabels-1)
    sum_area = sum_area/(nlabels-1)
    print(sum_h)
    print(sum_w)
    print(sum_area)
    

    for i in range(1, nlabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cx, cy) = centroids[i]
        # print(cx,cy)
        # print(w)
        # print(h)
        # print(area)
        # out = image.copy()
        # cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,0), 3)
        # cv2.circle(out, (int(cx), int(cy)), 4, (0,0,255), -1)
        # print(dim[0]/4)
        check_w = w >=  sum_w-15 and w <= sum_w+40
        check_h = h >= sum_h-5 and h <= sum_h+65
        check_area = area >= min(100, sum_area-50) and area < sum_area+1200
        check_y = cy > dim[0]/2-dim[0]/4 and cy < dim[0]/2+dim[0]/4
        check_x = cx >= 20 and cx <= dim[1]-20
        c = 0
        if all((check_area, check_h, check_w, check_y, check_x)):
            # print(c)
            # c=c+1
            c_mask = (labels == i).astype("uint8") * 255
            result = cv2.bitwise_or(result, c_mask)
        # c_mask= (labels==i).astype("uint8") * 255
        # cv2.imshow("out", out)
        # cv2.imshow("c_mask", c_mask)
        # cv2.waitKey(0)
    # kernel2 = np.ones((3,3), dtype= np.uint8)/25
    # result= cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel2)
    # result = cv2.GaussianBlur(result,(3,3),10,10)
    # cv2.imshow('image', image)
    cv2.imshow('result', result)
    cv2.imwrite('result.jpg', result)
    cv2.waitKey(0)
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    coords = []
    c = 0
    meanArea = 0
    for cnt in contours:
        meanArea = meanArea+cv2.contourArea(cnt)
    meanArea = meanArea/len(contours)
    # print(len(contours))
    # avg_h=0
    # for cnt in contours:
    #     (x,y,w,h)=cv2.boundingRect(cnt)
    #     avg_h=avg_h+h
    # avg_h=avg_h/len(contours)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        ratio = w/h
        # print(cv2.contourArea(cnt))
        if cv2.contourArea(cnt) > 0.20*meanArea:
            # print(ratio)
            if ratio > 2.4:
                half_width = int(w / 2)
                # print(half_width)
                coords.append((x, y, half_width, h))
                coords.append((x + half_width, y, half_width, h))
                c = c+2
            else:
                coords.append((x, y, w, h))
                c = c+1
    coords = sorted(coords, key=lambda x: x[0])
    # print(c)
    img_paths = []
    colored_paths = []
    for i in range(c):
        res = filter(result[coords[i][1]:coords[i][1]+coords[i][3], coords[i][0]:coords[i][0]+coords[i][2]])
        res2 = image[coords[i][1]:coords[i][1]+coords[i][3], coords[i][0]:coords[i][0]+coords[i][2]]
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        filename = 'char'+str(i)+'.png'
        file_name = 'character'+str(i)+'.png'
        cv2.imwrite(filename, res)
        cv2.imwrite(file_name, res2)
        img_paths.append(filename)
        colored_paths.append(file_name)
    return np.array(img_paths), np.array(colored_paths)


model = load_model('my_model2.h5')


def Plate_Recognition():
    # Enter filenames to be tested in image_paths after adding them to this folder
   
    image_paths = ['out2.jpg']
    
    for i in image_paths:
        image = cv2.imread(i)
        # image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_paths, colored_paths = extract_character(image)
        
        ans = []
        for i, j in zip(img_paths, colored_paths):
            
            img = cv2.imread(i)
            image = cv2.imread(j)
            image = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT)
            image = cv2.resize(image, (28, 28))
            # img = cv2.bitwise_not(img)
            cv2.imshow("img"+str(i), img)
            # cv2.imshow("image"+str(j), image)
            # cv2.waitKey(0)
            img_arr = np.asarray(img)
            img_arr = img_arr/255
            # img_arr.shape
            y = [img_arr]
            y = np.array(y)
            k = model.predict(y)
            df = pd.read_csv('keys.csv')
            df.drop(columns='Unnamed: 0', inplace=True)
            train_y = np.asarray(df)
            l_b = lb()
            Y = l_b.fit_transform(train_y)
            prediction = l_b.inverse_transform(k)
            pre = prediction[0]
            print(pre)
            ans.append(pre)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(ans)


if __name__ == '__main__':
    Plate_Recognition()
