# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
#import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
#import utils.cython_bbox
import cPickle
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg

class vehicles(imdb):
    def __init__(self, image_set, version, dataset_path=None):
        imdb.__init__(self, 'vehicles_dataset_v{}_{}'.format(version, image_set))
        self._version = version
        self._image_set = image_set
        self._dataset_path = self._get_default_path() if dataset_path is None \
                            else dataset_path
        self._annotation_path = os.path.join(self._dataset_path, 'annotations')
        self._image_path = os.path.join(self._dataset_path, 'images')
        self._raw_annotations = None
        self._classes = ('__background__', # always index 0
                         'car', 'van', 'truck',
                         'concretetruck', 'bus')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._dataset_path), \
                'Dataset path does not exist: {}'.format(self._dataset_path)
        assert os.path.exists(self._annotation_path), \
                'Path does not exist: {}'.format(self._annotation_path)
        assert os.path.exists(self._image_path), \
                'Path does not exist: {}'.format(self._image_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        image_path = os.path.join(self._image_path, self._image_index[i])
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_index = self._load_annotations().keys()
        return image_index

    def _get_default_path(self):
        """
        Get root of dataset directory
        """
        return os.path.join(cfg.DATA_DIR, 'vehicles_dataset_v{}'.format(self._version))

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

        raw_annotations = self._load_annotations()
        gt_roidb = self._format_raw_annotations(raw_annotations)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
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

    def _load_annotations(self):
        """
        Getter and setterof the unprocessed annotations of vehicles dataset.
        """
        if self._raw_annotations is not None:
            return self._raw_annotations

        dataset_file = os.path.join(self._annotation_path, 'complete_dataset_v{}.pkl'.format(self._version))
        idx_file = os.path.join(self._annotation_path, 'splits_indices_v{}.pkl'.format(self._version))

        def get_split_from_ds(ds, idx):
            split = {}
            keys = ds.keys()
            for j in xrange(len(idx)):
                k = keys[idx[j]]
                split[k] = ds[k]
            return split

        with open(idx_file, 'rb') as fid:
            indices = cPickle.load(fid)[self._image_set]
        with open(dataset_file, 'rb') as fid:
            ds = cPickle.load(fid)
            self._raw_annotations = get_split_from_ds(ds, indices)

        return self._raw_annotations

    def _format_raw_annotations(self, raw_annotations):
        annotations = []
        for j in xrange(self.num_images):
            anno = raw_annotations[self._image_index[j]]
            annotations.append(self._format_raw_annotation(anno))
        return annotations

    def _format_raw_annotation(self, annotation):
        num_objs = len(annotation['boxes'])
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for vehicles is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        boxes[:] = annotation['boxes']
        gt_classes[:] = np.array([self._class_to_ind[cls.lower()]
                                  for cls in annotation['gt_classes']])
        for ix, cls in enumerate(gt_classes):
            overlaps[ix, cls] = 1.0
            x1, y1, x2, y2 = boxes[ix]
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes' : gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _write_voc_results_file(self, all_boxes):
        """
        TODO: HOW TO USE THIS SCHNIZZ TO EVAL RESULTS
        """
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        """
        TODO: HOW TO USE THIS SCHNOZZ TO EVALUATE RESULTS
        """
        annopath = os.path.join(
            self._dataset_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._dataset_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._dataset_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir):
        """
        MAKE PYTHOIN EVAL WORK ON ALL BOXES PICKLE FILE
        """
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)

if __name__ == '__main__':
    d = vehicles('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
