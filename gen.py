# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import numpy as np
import h5py
import json
import os, traceback
import os.path as osp
from synthgen import *
from common import *


## Define some configuration variables:
NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1 # no. of times to use the same image
SECS_PER_IMG = 5 #max time per image in seconds

CONFIG = {'im_dir': '/home/czy/workspace/SynthText/data/bg_img',
          'depth_db': '/home/czy/workspace/SynthText/data/depth.h5',
          'seg_db': '/home/czy/workspace/SynthText/data/seg.h5',
          'out_dir': '/home/czy/workspace/SynthText/data/results',
          'data_dir': '/home/czy/workspace/SynthText/data/renderer_data'}

def add_res_to_db(imgname,res,db):
  """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
  ninstance = len(res)
  for i in range(ninstance):
    dname = "%s_%d"%(imgname, i)
    db['data'].create_dataset(dname,data=res[i]['img'])
    db['data'][dname].attrs['charBB'] = res[i]['charBB']
    db['data'][dname].attrs['wordBB'] = res[i]['wordBB']        
    #db['data'][dname].attrs['txt'] = res[i]['txt']
    L = res[i]['txt']
    L = [n.encode("ascii", "ignore") for n in L]
    db['data'][dname].attrs['txt'] = L

def save_image_and_labels(imname, res, output_dir):
    """
    Save the synthetically generated text image and its labels as a JSON file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, result in enumerate(res):
        img = result['img']
        img_path = osp.join(output_dir, f"{imname}_{i}.jpg")
        # 保存图片
        Image.fromarray(img).save(img_path)
        
        # 准备标签数据
        labels = {
            'charBB': result['charBB'].tolist(),
            'wordBB': result['wordBB'].tolist(),
            'txt': [text for text in result['txt']]
        }
        # 保存标签为 JSON 文件
        labels_path = osp.join(output_dir, f"{imname}_{i}.json")
        with open(labels_path, 'w') as f:
            json.dump(labels, f)

def main(viz=False, no_db=False):
  config = CONFIG
  
  if no_db:
    out_dir = config['out_dir'] + "/img"
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

  # open databases:
  print (colorize(Color.BLUE,'getting data..',bold=True))
  imdir = config['im_dir']
  outdir = config['out_dir']
  depth_db = h5py.File(config['depth_db'],'r')
  seg_db = h5py.File(config['seg_db'],'r')
  print (colorize(Color.BLUE,'\t-> done',bold=True))

  out_db = h5py.File(osp.join(outdir, 'SynthText.h5'),'w')
  out_db.create_group('/data')
  print (colorize(Color.GREEN,'Storing the output in: '+ config['out_dir'], bold=True))

  # # get the names of the image files in the dataset:
  imnames = sorted(depth_db.keys())
  N = len(imnames)
  global NUM_IMG
  if NUM_IMG < 0:
    NUM_IMG = N
  start_idx,end_idx = 0,min(NUM_IMG, N)

  RV3 = RendererV3(config['data_dir'],max_time=SECS_PER_IMG)
  for i in range(start_idx,end_idx):
    imname = imnames[i]
    try:
      img = Image.open(osp.join(imdir,imname))
      if img is None:
        continue
      mode = img.mode

      # get depth:
      depth = depth_db[imname][:].T
      depth = depth[:,:,0]

      # get segmentation:
      seg = seg_db['mask'][imname][:].astype('float32')
      area = seg_db['mask'][imname].attrs['area']
      label = seg_db['mask'][imname].attrs['label']

      # re-size uniformly:
      sz = depth.shape[:2][::-1]
      img = np.array(img.resize(sz,Image.ANTIALIAS))
      if mode == "RGBA":
         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
      seg = np.array(Image.fromarray(seg).resize(sz,Image.NEAREST))

      print(colorize(Color.RED,'%d of %d'%(i,end_idx-1), bold=True))
      print("img.shape", img.shape, osp.join(imdir,imname), mode)
      res = RV3.render_text(img,depth,seg,area,label,
                            ninstance=INSTANCE_PER_IMAGE,viz=viz)
      if len(res) > 0:
          if no_db:
            # 如果 no_db 为 True，则保存图片和标签
            save_image_and_labels(imname, res['src'], config["out_dir"] + "/src")
            save_image_and_labels(imname, res['tgt'], config["out_dir"] + "/tgt")
          else:
            # 如果 no_db 为 False，则添加到数据库
            add_res_to_db(imname, res, out_db)

      if viz:
          if 'q' in input(colorize(Color.RED,'continue?',True)):
              break

    except KeyboardInterrupt:
        print(colorize(Color.GREEN, '>>>> Stopping by user request.', bold=True))
        break
    except Exception as e:
        traceback.print_exc()
        print(colorize(Color.GREEN, '>>>> CONTINUING....', bold=True))
        continue

  depth_db.close()
  seg_db.close()
  out_db.close()


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations')
  parser.add_argument('--no_db', action='store_true', dest='no_db', default=False, help='flag to not store results in the database')
  args = parser.parse_args()
  main(args.viz, args.no_db)
