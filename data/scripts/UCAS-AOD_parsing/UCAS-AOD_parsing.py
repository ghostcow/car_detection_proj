import os.path as osp
import os
import numpy as np

ROOT = '/home/lioruzan/car_detection_proj/'
DATA = osp.join(ROOT,'data','UCAS-AOD')
NEG = osp.join(DATA,'Neg')
PLANE = osp.join(DATA,'PLANE')
CARS = osp.join(DATA,'CAR')

def main():
    ## load car annotations
    files=os.listdir(CARS)
    txts = sorted([f for f in files if osp.splitext(f)[1] == '.txt'])
    pngs = sorted([f for f in files if osp.splitext(f)[1] == '.png'])

    ## build car annotation db
    anno_db = {}
    for txtfile in txts:
        with open(osp.join(CARS,txtfile)) as f:
            txt = f.readlines()
        raw_arr = [line.split('\t')[-5:-1] for line in txt]
        anno = np.array(raw_arr,dtype=np.float32)

        name = osp.splitext(txtfile)[0]
        anno_db[name]=anno
    # print(anno_db)

if __name__ == '__main__':
    main()
