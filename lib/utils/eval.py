import numpy as np


# gets IoUs for a single pic
def get_ious(gt, dt):
    """
    output 
        -a Dx(G+1) grid of IoUs (D = num of detections, G = num of ground truths)
        -G+1'th columns are the score vector
    """ 
    ious = np.zeros((len(dt),len(gt)), dtype=np.float32)
    x1_dt, x1_gt = dt[:, 0], gt[:, 0]
    y1_dt, y1_gt = dt[:, 1], gt[:, 1]
    x2_dt, x2_gt = dt[:, 2], gt[:, 2]
    y2_dt, y2_gt = dt[:, 3], gt[:, 3]
    scores       = dt[:, 4][:, np.newaxis]
    
    areas_dt = (x2_dt - x1_dt + 1) * (y2_dt - y1_dt + 1)
    areas_gt = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)
    
    for i in xrange(len(dt)):
        xx1 = np.maximum(x1_dt[i], x1_gt)
        yy1 = np.maximum(y1_dt[i], y1_gt)
        xx2 = np.minimum(x2_dt[i], x2_gt)
        yy2 = np.minimum(y2_dt[i], y2_gt)
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        ovr = inter / (areas_dt[i] + areas_gt - inter)
        ious[i] = ovr
    return np.hstack([ious, scores])

def cap_detections_at_100(dt):
    for j in xrange(len(dt)):
        dets = dt[j]
        if len(dets) == 0: continue
        idx = np.argsort(dets[:, 4])[::-1][:100]
        dt[j] = dets[idx]
    return dt

def get_mean_accuracy(ious, iou_thres, recall_thres):

    accuracy = []
    for iou_threshold in iou_thres:

        iou_accuracy = []
        for recall_thresholds in recall_thres:

            recall_accuracy = []
            for rt in recall_thresholds:
                recall_accuracy.append(_get_accuracy(ious, iou_threshold, rt))

            iou_accuracy.append(np.mean(recall_accuracy))
        accuracy.append(iou_accuracy)
    return np.mean(accuracy)

def _get_accuracy(ious, iou_thres=0.5, score_thres=0.5):
    z, s = _get_z_s(ious, iou_thres, score_thres)
    accuracy = np.sum(np.dot(z, s)) / np.sum(s)
    return accuracy

def get_recall_thresholds(ious, iou_thresholds):
    return [_get_recall_threshold(ious, it) for it in iou_thresholds]

def _get_recall_threshold(ious, iou_thres):
    """
    return score thresholds for each distinct recall value at iou threshold iou_thres
    """
    score_thresholds = _get_score_thresholds(ious)
    N = sum([_iou.shape[1] - 1 for _iou in ious])
    thresholds = []
    prev_recall = -1
    for score_thres in reversed(score_thresholds):
        z, s = _get_z_s(ious, iou_thres=iou_thres, score_thres=score_thres)
        recall = np.sum(np.dot(z, s)) / N
        if recall > prev_recall:
            thresholds.insert(0, score_thres)
            prev_recall = recall
    return thresholds

def _get_score_thresholds(ious):
    """
    return all possible thresholds for detections
    @pre: ious must contain a fifth column of score per detection
    """
    raw_scores = np.hstack([_ious[:, -1] for _ious in ious])
    scores = np.sort(np.unique(raw_scores))
    thresholds = np.zeros(len(scores)-1)
    for j in xrange(len(thresholds)):
        cur_thresh = (scores[j]+scores[j+1])/2
        thresholds[j] = cur_thresh
    return thresholds

def _get_z_s(ious, iou_thres=0.5, score_thres=0.5):
    s = np.zeros((0,))
    z = np.zeros((0,))
    for i in xrange(len(ious)):
        D, G = ious[i].shape

        # sanity check
        for j in xrange(D):
            assert((ious[i][j, :-1] >= iou_thres).sum() <= 1)
        for j in xrange(G - 1):
            assert((ious[i][:, j] >= iou_thres).sum() <= 1)

        # build z vector
        zi = np.zeros(D)
        # fill zi with ones at locations where a match was found
        for j in xrange(D):
            idx = np.argmax(ious[i][j, :-1])    
            if ious[i][j, idx] >= iou_thres:
                zi[j] = 1
        z = np.hstack([z, zi])

        # build s vector
        # fill si with ones at locations above score threshold
        si = ious[i][:, -1] >= score_thres
        s = np.hstack([s, si])
    return z, s