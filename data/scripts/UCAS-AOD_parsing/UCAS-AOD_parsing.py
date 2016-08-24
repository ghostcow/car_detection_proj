import os.path as osp
import os
import numpy as np
import cPickle as pkl

ROOT = '/home/lioruzan/car_detection_proj/'
DATA = osp.join(ROOT,'data','UCAS-AOD')
NEG = osp.join(DATA,'Neg')
PLANE = osp.join(DATA,'PLANE')
CARS = osp.join(DATA,'CAR')

def xyhw_to_x1y1x2y2(boxes):
    xctr,yctr,width,height = boxes.T
    xctr = xctr[:,np.newaxis]
    yctr = yctr[:,np.newaxis]
    width = width[:,np.newaxis]
    height = height[:,np.newaxis]

    x1 = xctr - width*0.5
    y1 = yctr - height*0.5
    x2 = x1 + width
    y2 = y1 + height
    return np.concatenate(( x1,y1,x2,y2 ),axis=1)

def make_chinese_car_db():
    ## load car annotations
    files=os.listdir(CARS)
    txts = sorted([f for f in files if osp.splitext(f)[1] == '.txt'])
    pngs = sorted([f for f in files if osp.splitext(f)[1] == '.png'])

    ## build car annotation db
    anno_db = {}
    for idx,txtfile in enumerate(txts):
        with open(osp.join(CARS,txtfile)) as f:
            txt = f.readlines()
        raw_arr = np.array([line.split('\t')[-5:-1] for line in txt],dtype=np.float32)
        anno = xyhw_to_x1y1x2y2(raw_arr)
        gt_classes = ['Car' for row in anno]

        name = pngs[idx]
        anno_db[name]= {
            'boxes': anno,
            'gt_classes': gt_classes
        }
    # print(anno_db)
    return anno_db

# now it's time to add it to the splits. load the splits!
VEHICLES = osp.join(ROOT,'data','vehicles_dataset_v2','annotations')
FULL_DB = osp.join(VEHICLES, 'complete_dataset_v2.pkl')
SPLITS = osp.join(VEHICLES, 'splits_indices_v2.pkl')

def main():
    with open(FULL_DB) as f:
        full = pkl.load(f)
    with open(SPLITS) as f:
        splits = pkl.load(f)

    # join pics to old db
    new_cars = make_chinese_car_db()
    new_full = full.copy()
    new_full.update(new_cars)

    # update split indices
    _full = full.keys()
    _new_full = new_full.keys()
    old2new = {}
    for idx,name in enumerate(_full):
        new_idx = _new_full.index(name)
        old2new[idx]=new_idx

    for k,v in splits.iteritems():
        for i,idx in enumerate(v):
            v[i]=old2new[idx]

    # update train split with new indices
    _new_cars = np.array([new_full.keys().index(o) for o in new_cars.keys()])
    splits['train'] = np.concatenate((splits['train'],_new_cars))
    np.random.shuffle(splits['train'])

    # save dat shite in version 3 fldr
    with open(FULL_DB.replace('v2','v3'), 'wb') as f:
        pkl.dump(new_full,f)
    with open(SPLITS.replace('v2','v3'), 'wb') as f:
        pkl.dump(splits,f)

if __name__ == '__main__':
    main()
