# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.car_ds
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
from utils.eval import *
import cPickle
import subprocess

class car_ds(datasets.imdb):
    def __init__(self, image_set, devkit_path=None):
        datasets.imdb.__init__(self, 'cars_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._classes = ('__background__', # always index 0
                         'car')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000,
                       'use_diff' : False,
                       'rpn_file' : None}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._devkit_path), \
                'Path does not exist: {}'.format(self._devkit_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._devkit_path, index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._devkit_path,
                                      'OIRDS_{}.pkl'.format(self._image_set))
        assert(os.path.exists(image_set_file))
        with open(image_set_file, 'rb') as f:
            image_index = cPickle.load(f).keys()
        # sort image_index to match SS list order
        return sorted(image_index)

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'OIRDS_v1_0')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_car_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['all_boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [obj for obj in objs
                             if int(get_data_from_tag(obj, 'difficult')) == 0]
            if len(non_diff_objs) != len(objs):
                print 'Removed {} difficult objects' \
                    .format(len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = float(get_data_from_tag(obj, 'xmin')) - 1
            y1 = float(get_data_from_tag(obj, 'ymin')) - 1
            x2 = float(get_data_from_tag(obj, 'xmax')) - 1
            y2 = float(get_data_from_tag(obj, 'ymax')) - 1
            cls = self._class_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _load_car_annotation(self, index):
        """
        Load image and bounding boxes info from pickled files of Car Dataset
        """
        cache_file = os.path.join(self._devkit_path, 'OIRDS_{}.pkl'.format(self._image_set))
        assert(os.path.exists(cache_file))
        with open(cache_file, 'rb') as f:
            ds = cPickle.load(f)
        objs = ds[index]  # list of bboxes
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame
        for ix, box in enumerate(objs):
            x1 = float(box[0])
            y1 = float(box[1])
            x2 = float(box[2])
            y2 = float(box[3])
            # TODO: edit here after adding labels to ds
            cls = 1
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped' : False}

    def evaluate_detections(self, all_boxes, output_dir):
        """
        Implemented MS COCO evaluation algorithm to evaluate detection results
        """
        import string
        import random
        import time

        with open('data/OIRDS_v1_0/OIRDS_{}.pkl'.format(self._image_set), 'rb') as f:
            raw_gt = cPickle.load(f)
            pics = sorted(raw_gt.keys())
            gt = [np.array(raw_gt[k]) for k in pics]
        # num of dets per image must be <= 100
        dt = cap_detections_at_100([np.array(b) for b in all_boxes[1]])

        iou_thres = [0.50]
        t0 = time.time()
        ious = [get_ious(_gt, _dt) for _gt, _dt in zip(gt, dt) if _dt.shape[0] != 0]
        recall_thres, recalls = zip(*get_recall_thresholds(ious, iou_thres))
        mean_accuracy = get_mean_accuracy(ious, iou_thres, recall_thres)
        average_recall = np.mean(np.array(recalls))
        results = 'mAP={:.02f}, recall={} at IoU thresholds {}, in time {:.02f}'.format(mean_accuracy * 100, average_recall * 100, iou_thres, time.time()-t0)

        with open(os.path.join(output_dir, 'results1.log'), 'w') as f:
            f.write(results + '\n')
        print(results)

        iou_thres = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        t0 = time.time()
        ious = [get_ious(_gt, _dt) for _gt, _dt in zip(gt, dt) if _dt.shape[0] != 0]
        recall_thres, recalls = zip(*get_recall_thresholds(ious, iou_thres))
        mean_accuracy = get_mean_accuracy(ious, iou_thres, recall_thres)
        average_recall = np.mean(np.array(recalls))
        results = 'mAP={:.02f} at IoU thresholds {}, in time {:.02f}'.format(mean_accuracy * 100, average_recall * 100, iou_thres, time.time()-t0)

        with open(os.path.join(output_dir, 'results2.log'), 'w') as f:
            f.write(results + '\n')
        print(results)
        
        with open(os.path.join(output_dir, 'nms_detections.pkl'), 'wb') as f:
            cPickle.dump(all_boxes, f)
        

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.car_ds('train')
    res = d.roidb
    from IPython import embed; embed()
