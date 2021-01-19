
import cv2
import numpy as np
from mean_average_precision import MetricBuilder
import sys
import os
import pandas as pd
import warnings
from tqdm import tqdm

def syntaxis():
    """ Use command """
    print("\nUse: \tpython3 evaluate.py GT_DIR PREDICT_DIR\n")
    print("GT_DIR: \tdirectory containing the ground-truth provided (txt + jpg).")
    print("PREDICT_FILE: \tfile containing the predicted bounding boxes.")

def IoU(boxA, boxB):
    """
    Computes the Intersection over Union (IoU) metric, given
    two bounding boxes.
    Input:
    "boxA": bounding box A
    "boxB": bounding box B
    Output:
    "score": IoU score
    """

    # compute the intersection points of the two BBs
    xLeft = max(boxA[0], boxB[0])
    yLeft = max(boxA[1], boxB[1])
    xRight = min(boxA[2], boxB[2])
    yRight = min(boxA[3], boxB[3])

    # compute the area of the intersection rectangle
    interArea = max(0, xRight - xLeft + 1) * max(0, yRight - yLeft + 1)    

    # compute the area of both boxA and boxB rectangles
    boxA_area = (boxA[2]-boxA[0] + 1)*(boxA[3]-boxA[1] + 1)
    boxB_area = (boxB[2]-boxB[0] + 1)*(boxB[3]-boxB[1] + 1)

    # compute the intersection over union
    score = interArea / float(boxA_area + boxB_area - interArea)

    return score

def GIoU(boxA, boxB):
    """
    Computes the Generalized Intersection over Union (GIoU) metric, given
    two bounding boxes.
    Input:
    "boxA": bounding box A
    "boxB": bounding box B
    Output:
    "score": IoU score
    """

    # compute the intersection points of the two BBs
    xLeft = max(boxA[0], boxB[0])
    yLeft = max(boxA[1], boxB[1])
    xRight = min(boxA[2], boxB[2])
    yRight = min(boxA[3], boxB[3])

    # compute the area of the intersection rectangle
    interArea = max(0, xRight - xLeft + 1) * max(0, yRight - yLeft + 1)    

    # compute the area of both boxA and boxB rectangles
    boxA_area = (boxA[2]-boxA[0] + 1)*(boxA[3]-boxA[1] + 1)
    boxB_area = (boxB[2]-boxB[0] + 1)*(boxB[3]-boxB[1] + 1)

    unionArea = boxA_area + boxB_area - interArea
    # compute the intersection over union
    IoU = interArea / float(unionArea)

    # compute the coordinates of the smallest enclosion box
    x_min = min(boxA[0],boxB[0])
    y_min = min(boxA[1],boxB[1])
    x_max = max(boxA[2], boxB[2])
    y_max = max(boxA[3], boxB[3])

    # compute the area of the smallest enclosion box
    boxC_area = (x_max - x_min + 1)*(y_max - y_min +1)

    # compute the generalized intersection over union GIoU
    score = IoU - (boxC_area - unionArea)/float(boxC_area)

    return score

def NMS(BBs, thresh):
    """
    Computes the Non-maximum Suppression algorithm, removing overlapping
    predicted bounding boxes, while keeping the ones with the highest
    confidence score.
    """
    D = []    
    sorted_confidence = list(BBs[(-BBs[:,5]).argsort()])

    # Repeat until there are no more proposals left in BBs
    while len(sorted_confidence) > 0:
        # 1. Select the proposal with highest confidence score, remove it from BBs
        # and add it to the final proposal list D. (Initially D is empty).
        best = sorted_confidence.pop(0)
        D.append(best)

        # 2. Compare this proposal with all the proposals — calculate the IOU
        # of this proposal with every other proposal. If the IOU is greater than
        # the threshold, remove that proposal from BBs.
        to_remove = []
        for i in range(len(sorted_confidence)):
            if IoU(best, sorted_confidence[i]) > thresh:
                to_remove.append(i)
        
        for i in sorted(to_remove, reverse=True):
            sorted_confidence.pop(i)
    
    return np.array(D)

def load_predicted(path):
    """
    Loads the bounding boxes of file "path"
    Input:
    "path": file containing one bounding box per line. 
    Output:
    "pages": dictionary with key = page_name, value = array containing
    the predicted bounding boxes    
    """   
    rows = pd.read_csv(path, header=None).to_numpy()
    pages = {}

    with tqdm(desc="Processing '" + path + "':", total=len(rows), ascii=True, bar_format='{desc} |{bar:10}| {percentage:3.0f}%') as pbar:
        for r in rows:
            # Remove the bounding boxes that have a confidence score of less than 0.05
            if r[-2] >= 0.05:
                page_name = r[0]
                list_BBs = pages.get(page_name)
                
                # [xmin, ymin, xmax, ymax, class_id, confidence]
                if list_BBs is None:    
                    list_BBs = np.array([[int(r[1]),int(r[2]),int(r[3]),int(r[4]),int(r[6]),r[5]]])
                else:
                    newrow = np.array([int(r[1]),int(r[2]),int(r[3]),int(r[4]),int(r[6]),r[5]])
                    list_BBs = np.vstack([list_BBs, newrow])

                pages[page_name] = list_BBs
                pbar.update(1)
    
    return pages

def load_gt(path):
    """
    Loads the bounding boxes of directory "path"
    Input:
    "path": 
    Output:
    "pages": dictionary with key = page_name, value = array containing
    the ground truth bounding boxes
    """
    pages = {}

    with tqdm(desc="Processing GT:", total=len(os.listdir(path))/2, ascii=True, bar_format='{desc} |{bar:10}| {percentage:3.0f}%') as pbar:
        for f in os.listdir(path):
            if f.endswith("txt"):
                img_name = f.replace("color_","").replace("txt","jpg")
                full_path = os.path.join(path, img_name)
                img = cv2.imread(full_path)

                h_img, w_img = img.shape[:2]    
                
                # read the lines of the gt file
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    line = np.genfromtxt(os.path.join(path,f), usecols=(0,1,2,3,4),comments='#',)
                    if len(line) == 0:
                        pages[img_name] = np.array([])
                    else:
                        if len(line.shape) == 1: line = [line]            
                        for l in line:     
                            x = int(l[0] * w_img / 100)
                            y = int(l[1] * h_img / 100)
                            w = int(l[2] * w_img / 100)
                            h = int(l[3] * h_img / 100)
                            label = int(l[4])

                            # dictionary key = page img name
                            list_BBs = pages.get(img_name)

                            # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
                            if list_BBs is None:    
                                list_BBs = np.array([[x, y, x+w-1, y+h-1, label, 0, 0]])
                            else:
                                newrow = np.array([x, y, x+w-1, y+h-1, label, 0, 0])
                                list_BBs = np.vstack([list_BBs, newrow])

                            pages[img_name] = list_BBs
                pbar.update(1)
            
    return pages

def draw_IoUs(pred, gt, img_path):    
    """
    Draw the bounding boxes of predicted and ground-truth
    bounding boxes and the associated Intersection over Union (IoU)
    score.
    Input:
    """

    results_path = "Results"
    os.makedirs(results_path,exist_ok=True)

    with tqdm(desc="Generating Results:", total=len(gt.keys()), ascii=True, bar_format='{desc} |{bar:10}| {percentage:3.0f}%') as pbar:
        for key in gt.keys():
            gt_bbs = gt.get(key)
            pred_bbs = pred.get(key, np.array([]))

            # ground-truth image
            img = cv2.imread(os.path.join(img_path, key))
            h, w = img.shape[:2]

            tagged_pred_IoU = set()
            tagged_pred_GIoU = set()
            for bb in gt_bbs:
                # calculate the best IoU score for a given ground truth bounding box
                IoU_scores = np.array([IoU(bb, pred) for pred in pred_bbs])
                best_IoU = np.argmax(IoU_scores) if len(IoU_scores) > 0 else None
                if best_IoU is not None: tagged_pred_IoU.add(best_IoU)
                score_IoU = IoU_scores[best_IoU] if best_IoU is not None else 0

                # calculate the best GIoU score for a given ground truth bounding box
                GIoU_scores = np.array([GIoU(bb, pred) for pred in pred_bbs])
                best_GIoU = np.argmax(GIoU_scores) if len(GIoU_scores) > 0 else None
                if best_GIoU is not None: tagged_pred_GIoU.add(best_GIoU)
                score_GIoU = GIoU_scores[best_GIoU] if best_GIoU is not None else 0                

                # draw the ground-truth bounding boxes
                x_min = bb[0]
                y_min = bb[1]
                x_max = bb[2]
                y_max = bb[3]
                
                cv2.rectangle(img, (x_min,y_min), (x_max,y_max), (0,255,0), 2)

                # write the corresponding IoU score
                cv2.putText(img, "{:.4f}".format(score_IoU), (int((x_max - x_min)/2 + x_min - int(0.024*w)), y_min-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                
                # write the corresponding GIoU score
                cv2.putText(img, "{:.4f}".format(score_GIoU), (int((x_max - x_min)/2 + x_min - int(0.024*w)), y_min-int(0.01*h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
                

            for bb in range(len(pred_bbs)):
                x_min = int(pred_bbs[bb][0])
                y_min = int(pred_bbs[bb][1])
                x_max = int(pred_bbs[bb][2])
                y_max = int(pred_bbs[bb][3])

                # draw the predicted bounding boxes
                cv2.rectangle(img, (x_min,y_min), (x_max,y_max), (0,0,255), 2)

                if bb not in tagged_pred_IoU:
                    # write a IoU score of 0
                    cv2.putText(img, "0.000", (int((x_max - x_min)/2 + x_min - int(0.024*w)), y_min-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                
                if bb not in tagged_pred_GIoU:
                    # write a GIoU score of 0
                    cv2.putText(img, "0.000", (int((x_max - x_min)/2 + x_min - int(0.024*w)), y_min-int(0.01*h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

            
            cv2.imwrite(os.path.join(results_path, key), img)    
            pbar.update(1)
    

if __name__ == '__main__':

    if len(sys.argv) != 3:
        syntaxis()
        exit()

    gt_path = sys.argv[1]
    pred_path = sys.argv[2]    

    # Check if the argument directories are valid
    if not os.path.isfile(pred_path):
        print(pred_path + " is not a valid file.")
        exit()    
    if not os.path.isdir(gt_path):
        print(gt_path + " is not a valid directory.")
        exit()

    # dictionary: key = page_img_name, value = list of predicted BBs
    pred_dict = load_predicted(pred_path)

    # remove overlapping predicted bounding boxes
    for key in pred_dict.keys():
        pred_dict[key] = NMS(pred_dict[key], 0.1)
        
    # dictionary: key = page_img_name, value = list of ground-truth BBs
    gt_dict = load_gt(gt_path)
    
    # draws the predicted and ground-truth bounding boxes and the resulting IoU scores
    draw_IoUs(pred_dict, gt_dict, gt_path)

    # create metric_fn
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", num_classes=2)
        
    with tqdm(desc="Calculating metric:'", total=len(gt_dict.keys()), ascii=True, bar_format='{desc} |{bar:10}| {percentage:3.0f}%') as pbar:
        for key in gt_dict.keys():
            preds = pred_dict.get(key, np.array([]))
            gt = gt_dict.get(key)        

            # add samples to evaluation
            if len(gt) > 0 and len(preds) > 0:
                metric_fn.add(preds, gt)
            # elif len(gt) == 0 and len(preds) > 0:
            #     print("predicted, gt empty:", key, len(preds))
            # elif len(gt) > 0 and len(preds) == 0:
            #     print("gt, predicted empty:", key, len(gt))

            
            pbar.update(1)

    # # compute PASCAL VOC metric
    print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")

    # compute PASCAL VOC metric at the all points
    print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")

    # compute metric COCO metric
    # print(f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")

         

        
    
