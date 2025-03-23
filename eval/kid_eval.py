import numpy as np 
import os 
from os import path
import shutil, subprocess
from PIL import Image
from PIL import ImageFile

def KID_score(original_dir, generated_dir):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    origin_pth_dir = original_dir
    trg_pth_dir = generated_dir
    
    npy_file = 'npy_file'
    
    #.npy 삭제
    if os.path.isfile(f'{origin_pth_dir}/{npy_file}.npy'):
        os.unlink(f'{origin_pth_dir}/{npy_file}.npy')
    if os.path.isfile(f'{trg_pth_dir}/{npy_file}.npy'):
        os.unlink(f'{trg_pth_dir}/{npy_file}.npy')
    
    original_lst = os.listdir(origin_pth_dir)
    original_imgs = [file for file in original_lst if not file.endswith(".npy")]
    
    generated_lst = os.listdir(trg_pth_dir)        
    generated_imgs = [file for file in generated_lst if not file.endswith(".npy")]
    
    if not os.path.isfile(f'{origin_pth_dir}/{npy_file}.npy'):

        img_list_np = []
        for i in original_imgs:
            img = Image.open(origin_pth_dir + '/' + i)
            img = img.convert('RGB')
            img = img.resize((384,512))
            img_array = np.asarray(img)
            img_list_np.append(img_array)
        img_np = np.array(img_list_np).astype("float64")
        img_np = img_np.transpose(0,3,1,2)
        print('original shape: ',np.shape(img_np))
        np.save(f'{origin_pth_dir}/{npy_file}',img_np)
    
    if not os.path.isfile(f'{trg_pth_dir}/{npy_file}.npy'):
        img_list_np = []
        
        for i in generated_imgs:
            img = Image.open(trg_pth_dir + '/' + i)
            img = img.convert('RGB')
            img = img.resize((384,512))
            img_array = np.asarray(img)
            img_list_np.append(img_array)
        img_np = np.array(img_list_np).astype("float64")
        img_np = img_np.transpose(0,3,1,2)
        print('generated shape: ',np.shape(img_np))
        np.save(f'{trg_pth_dir}/{npy_file}',img_np)
    
    proc = subprocess.Popen([f"python ./gan-metrics-pytorch/kid_score.py --true {origin_pth_dir}/{npy_file}.npy --fake {trg_pth_dir}/{npy_file}.npy --gpu 0"],  shell=True)
    _ , _ = proc.communicate()
    if os.path.isfile(f'{origin_pth_dir}/{npy_file}.npy'):
        os.unlink(f'{origin_pth_dir}/{npy_file}.npy')
    if os.path.isfile(f'{trg_pth_dir}/{npy_file}.npy'):
        os.unlink(f'{trg_pth_dir}/{npy_file}.npy')
    
if __name__=='__main__':
    
    generated_dir_lst = ['./StableVITON/results/unpair', 
                         './StableVITON/results/repaint/unpair',
                         './HR-VITON/Output/unpaired',
                         './GP-VTON/sample/test_gpvtongen_vitonhd_unpaired_1109']
    
    original_dir = './HR-VITON/data/zalando-hd-resized/test/image'
    generated_dir = generated_dir_lst[3]
    
    KID_score(original_dir, generated_dir)