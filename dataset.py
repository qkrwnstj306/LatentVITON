import os
from os.path import join as opj

import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

def imread(
        p, h, w, 
        is_mask=False, 
        in_inverse_mask=False, 
        img=None
):
    if img is None:
        img = cv2.imread(p)
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w,h))
        img = (img.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w,h))
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:,:,None]
        if in_inverse_mask:
            img = 1-img
    return img

def imread_for_albu(
        p, 
        is_mask=False, 
        in_inverse_mask=False, 
        cloth_mask_check=False, 
        use_resize=False, 
        height=512, 
        width=384,
):
    img = cv2.imread(p)
    if use_resize:
        img = cv2.resize(img, (width, height))
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img>=128).astype(np.float32)
        if cloth_mask_check:
            if img.sum() < 30720*4:
                img = np.ones_like(img).astype(np.float32)
        if in_inverse_mask:
            img = 1 - img
        img = np.uint8(img*255.0)
    return img
def norm_for_albu(img, is_mask=False):
    if not is_mask:
        img = (img.astype(np.float32)/127.5) - 1.0
    else:
        img = img.astype(np.float32) / 255.0
        img = img[:,:,None]
    return img

class VITONHDDataset(Dataset):
    def __init__(
            self, 
            data_root_dir, 
            img_H, 
            img_W, 
            is_paired=True, 
            is_test=False, 
            is_sorted=False, 
            transform_size=None, 
            transform_color=None,
            **kwargs
        ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.pair_key = "paired" if is_paired else "unpaired"
        self.data_type = "train" if not is_test else "test"
        self.is_test = is_test
        self.resize_ratio_H = 1.0
        self.resize_ratio_W = 1.0

        self.resize_transform = A.Resize(img_H, img_W)
        self.transform_size = None
        self.transform_crop_person = None
        self.transform_crop_cloth = None
        self.transform_color = None

        #### spatial aug >>>>
        transform_crop_person_lst = []
        transform_crop_cloth_lst = []
        transform_size_lst = [A.Resize(int(img_H*self.resize_ratio_H), int(img_W*self.resize_ratio_W))]
    
        if transform_size is not None:
            if "hflip" in transform_size:
                transform_size_lst.append(A.HorizontalFlip(p=0.5))

            if "shiftscale" in transform_size:
                transform_crop_person_lst.append(A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0))
                transform_crop_cloth_lst.append(A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0))

        self.transform_crop_person = A.Compose(
                transform_crop_person_lst,
                additional_targets={"agn":"image", 
                                    "agn_mask":"image", 
                                    "cloth_mask_warped":"image", 
                                    "cloth_warped":"image", 
                                    "image_densepose":"image", 
                                    "image_parse":"image", 
                                    "gt_cloth_warped_mask":"image", 
                                    "wrp_cloth":"image",
                                    "wrp_cloth_mask":"image",
                                    }
        )
        self.transform_crop_cloth = A.Compose(
                transform_crop_cloth_lst,
                additional_targets={"cloth_mask":"image"}
        )

        self.transform_size = A.Compose(
                transform_size_lst,
                additional_targets={"agn":"image", 
                                    "agn_mask":"image", 
                                    "cloth":"image", 
                                    "cloth_mask":"image", 
                                    "cloth_mask_warped":"image", 
                                    "cloth_warped":"image", 
                                    "image_densepose":"image", 
                                    "image_parse":"image", 
                                    "gt_cloth_warped_mask":"image",
                                    "wrp_cloth":"image",
                                    "wrp_cloth_mask":"image",
                                    }
            )
        #### spatial aug <<<<

        #### non-spatial aug >>>>
        if transform_color is not None:
            transform_color_lst = []
            for t in transform_color:
                if t == "hsv":
                    transform_color_lst.append(A.HueSaturationValue(5,5,5,p=0.5))
                elif t == "bright_contrast":
                    transform_color_lst.append(A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.02), contrast_limit=(-0.3, 0.3), p=0.5))

            self.transform_color = A.Compose(
                transform_color_lst,
                additional_targets={"agn":"image", 
                                    "cloth":"image",  
                                    "cloth_warped":"image",
                                    "wrp_cloth":"image",
                                    }
            )
        #### non-spatial aug <<<<
                    
        assert not (self.data_type == "train" and self.pair_key == "unpaired"), f"train must use paired dataset"
        
        im_names = []
        c_names = []
        with open(opj(self.drd, f"{self.data_type}_pairs.txt"), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                if im_name == "00126_00.jpg": # 00126_00
                    im_names.append(im_name)
                    c_names.append(c_name)
        if is_sorted:
            im_names, c_names = zip(*sorted(zip(im_names, c_names)))
        self.im_names = im_names
        
        self.c_names = dict()
        self.c_names["paired"] = im_names
        self.c_names["unpaired"] = c_names

    def __len__(self):
        return len(self.im_names)
    
    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]

        if self.transform_size is None and self.transform_color is None:
            agn = imread(
                opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]), 
                self.img_H, 
                self.img_W
            )
            agn_mask = imread(
                opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")), 
                self.img_H, 
                self.img_W, 
                is_mask=True, 
                in_inverse_mask=True
            )
            cloth = imread(
                opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]), 
                self.img_H, 
                self.img_W
            )
            cloth_mask = imread(
                opj(self.drd, self.data_type, "cloth-mask", self.c_names[self.pair_key][idx]), 
                self.img_H, 
                self.img_W, 
                is_mask=True, 
                cloth_mask_check=True
            )
            
            gt_cloth_warped_mask = imread(
                opj(self.drd, self.data_type, "gt_cloth_warped_mask", self.im_names[idx]), 
                self.img_H, 
                self.img_W, 
                is_mask=True
            ) if not self.is_test else np.zeros_like(agn_mask)

            image = imread(opj(self.drd, self.data_type, "image", self.im_names[idx]), self.img_H, self.img_W)
            image_densepose = imread(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]), self.img_H, self.img_W)

            """dummy img"""
            wrp_name = "cloth-warp" if self.pair_key == "paired" else "unpaired-cloth-warp"
            wrp_mask_name = "cloth-warp-mask" if self.pair_key == "paired" else "unpaired-cloth-warp-mask"
            wrp_cloth = imread_for_albu(opj(self.drd, self.data_type, wrp_name, self.c_names[self.pair_key][idx])) if self.is_test else np.zeros_like(image)
            wrp_cloth_mask = imread_for_albu(opj(self.drd, self.data_type, wrp_mask_name, self.c_names[self.pair_key][idx]), is_mask=True) if self.is_test else np.zeros_like(image)

        else:
            agn = imread_for_albu(opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]))
            agn_mask = imread_for_albu(opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")), is_mask=True)
            cloth = imread_for_albu(opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]))
            cloth_mask = imread_for_albu(opj(self.drd, self.data_type, "cloth-mask", self.c_names[self.pair_key][idx]), is_mask=True, cloth_mask_check=True)
            
            gt_cloth_warped_mask = imread_for_albu(
                opj(self.drd, self.data_type, "gt_cloth_warped_mask", self.im_names[idx]),
                is_mask=True
            ) if not self.is_test else np.zeros_like(agn_mask)
                
            image = imread_for_albu(opj(self.drd, self.data_type, "image", self.im_names[idx]))
            image_densepose = imread_for_albu(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]))

            """dummy img"""
            wrp_name = "cloth-warp" if self.pair_key == "paired" else "unpaired-cloth-warp"
            wrp_mask_name = "cloth-warp-mask" if self.pair_key == "paired" else "unpaired-cloth-warp-mask"
            wrp_cloth = imread_for_albu(opj(self.drd, self.data_type, wrp_name, self.c_names[self.pair_key][idx])) if self.is_test else np.zeros_like(image)
            wrp_cloth_mask = imread_for_albu(opj(self.drd, self.data_type, wrp_mask_name, self.c_names[self.pair_key][idx]), is_mask=True) if self.is_test else np.zeros_like(image)

            if self.transform_size is not None:
                transformed = self.transform_size(
                    image=image, 
                    agn=agn, 
                    agn_mask=agn_mask, 
                    cloth=cloth, 
                    cloth_mask=cloth_mask, 
                    image_densepose=image_densepose,
                    gt_cloth_warped_mask=gt_cloth_warped_mask,
                )
                image=transformed["image"]
                agn=transformed["agn"]
                agn_mask=transformed["agn_mask"]
                image_densepose=transformed["image_densepose"]
                gt_cloth_warped_mask=transformed["gt_cloth_warped_mask"]

                cloth=transformed["cloth"]
                cloth_mask=transformed["cloth_mask"]
                
            if self.transform_crop_person is not None:
                transformed_image = self.transform_crop_person(
                    image=image,
                    agn=agn,
                    agn_mask=agn_mask,
                    image_densepose=image_densepose,
                    gt_cloth_warped_mask=gt_cloth_warped_mask,
                    wrp_cloth=wrp_cloth,
                    wrp_cloth_mask=wrp_cloth_mask,
                )

                image=transformed_image["image"]
                agn=transformed_image["agn"]
                agn_mask=transformed_image["agn_mask"]
                image_densepose=transformed_image["image_densepose"]
                gt_cloth_warped_mask=transformed_image["gt_cloth_warped_mask"]

                wrp_cloth=transformed_image["wrp_cloth"]
                wrp_cloth_mask=transformed_image["wrp_cloth_mask"]

            if self.transform_crop_cloth is not None:
                transformed_cloth = self.transform_crop_cloth(
                    image=cloth,
                    cloth_mask=cloth_mask
                )

                cloth=transformed_cloth["image"]
                cloth_mask=transformed_cloth["cloth_mask"]

            agn_mask = 255 - agn_mask
            if self.transform_color is not None:
                transformed = self.transform_color(
                    image=image, 
                    agn=agn, 
                    cloth=cloth,
                    wrp_cloth=wrp_cloth,
                )

                image=transformed["image"]
                agn=transformed["agn"]
                cloth=transformed["cloth"]
                
                wrp_cloth=transformed["wrp_cloth"]

                agn = agn * agn_mask[:,:,None].astype(np.float32)/255.0 + 128 * (1 - agn_mask[:,:,None].astype(np.float32)/255.0)
                
            agn = norm_for_albu(agn)
            agn_mask = norm_for_albu(agn_mask, is_mask=True)
            cloth = norm_for_albu(cloth)
            cloth_mask = norm_for_albu(cloth_mask, is_mask=True)
            image = norm_for_albu(image)
            image_densepose = norm_for_albu(image_densepose)
            gt_cloth_warped_mask = norm_for_albu(gt_cloth_warped_mask, is_mask=True)

            wrp_cloth = norm_for_albu(gt_cloth_warped_mask)
            wrp_cloth_mask = norm_for_albu(wrp_cloth_mask, is_mask=True)

        return dict(
            agn=agn,
            agn_mask=agn_mask,
            cloth=cloth,
            cloth_mask=cloth_mask,
            image=image,
            image_densepose=image_densepose,
            gt_cloth_warped_mask=gt_cloth_warped_mask,
            wrp_cloth=wrp_cloth,
            wrp_cloth_mask=wrp_cloth_mask,
            txt="",
            img_fn=img_fn,
            cloth_fn=cloth_fn,
        )
    


class DresscodeDataset(Dataset):
    def __init__(
            self, 
            data_root_dir, 
            img_H, 
            img_W, 
            is_paired=True, 
            is_test=False, 
            is_sorted=False, 
            transform_size=None, 
            transform_color=None,
            **kwargs
        ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.pair_key = "paired" if is_paired else "unpaired"
        self.data_type = "train" if not is_test else "test"
        self.is_test = is_test
        self.resize_ratio_H = 1.0
        self.resize_ratio_W = 1.0

        self.resize_transform = A.Resize(img_H, img_W)
        self.transform_size = None
        self.transform_crop_person = None
        self.transform_crop_cloth = None
        self.transform_color = None

        #### spatial aug >>>>
        transform_crop_person_lst = []
        transform_crop_cloth_lst = []
        transform_size_lst = [A.Resize(int(img_H*self.resize_ratio_H), int(img_W*self.resize_ratio_W))]
    
        if transform_size is not None:
            if "hflip" in transform_size:
                transform_size_lst.append(A.HorizontalFlip(p=0.5))

            if "shiftscale" in transform_size:
                transform_crop_person_lst.append(A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0))
                transform_crop_cloth_lst.append(A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0))

        self.transform_crop_person = A.Compose(
                transform_crop_person_lst,
                additional_targets={"agn":"image", 
                                    "agn_mask":"image", 
                                    "cloth_mask_warped":"image", 
                                    "cloth_warped":"image", 
                                    "image_densepose":"image", 
                                    "image_parse":"image", 
                                    "gt_cloth_warped_mask":"image", 
                                    "wrp_cloth":"image",
                                    "wrp_cloth_mask":"image",
                                    }
        )
        self.transform_crop_cloth = A.Compose(
                transform_crop_cloth_lst,
                additional_targets={"cloth_mask":"image"}
        )

        self.transform_size = A.Compose(
                transform_size_lst,
                additional_targets={"agn":"image", 
                                    "agn_mask":"image", 
                                    "cloth":"image", 
                                    "cloth_mask":"image", 
                                    "cloth_mask_warped":"image", 
                                    "cloth_warped":"image", 
                                    "image_densepose":"image", 
                                    "image_parse":"image", 
                                    "gt_cloth_warped_mask":"image",
                                    "wrp_cloth":"image",
                                    "wrp_cloth_mask":"image",
                                    }
            )
        #### spatial aug <<<<

        #### non-spatial aug >>>>
        if transform_color is not None:
            transform_color_lst = []
            for t in transform_color:
                if t == "hsv":
                    transform_color_lst.append(A.HueSaturationValue(5,5,5,p=0.5))
                elif t == "bright_contrast":
                    transform_color_lst.append(A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.02), contrast_limit=(-0.3, 0.3), p=0.5))

            self.transform_color = A.Compose(
                transform_color_lst,
                additional_targets={"agn":"image", 
                                    "cloth":"image",  
                                    "cloth_warped":"image",
                                    "wrp_cloth":"image",
                                    }
            )
        #### non-spatial aug <<<<
                    
        assert not (self.data_type == "train" and self.pair_key == "unpaired"), f"train must use paired dataset"
        
        im_names = []
        c_names = []
        with open(opj(self.drd, f"{self.data_type}_pairs.txt"), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)
        if is_sorted:
            im_names, c_names = zip(*sorted(zip(im_names, c_names)))
        self.im_names = im_names
        
        self.c_names = dict()
        self.c_names["paired"] = im_names
        self.c_names["unpaired"] = c_names

    def __len__(self):
        return len(self.im_names)
    
    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]

        if self.transform_size is None and self.transform_color is None:
            agn = imread(
                opj(self.drd, "agnostic-v3.2", self.im_names[idx]), 
                self.img_H, 
                self.img_W
            )
            agn_mask = imread(
                opj(self.drd, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")), 
                self.img_H, 
                self.img_W, 
                is_mask=True, 
                in_inverse_mask=True
            )
            
            cloth_name = self.c_names[self.pair_key][idx]

            # cloth
            if self.pair_key == "paired":
                cloth_filename = cloth_name.replace("_0.jpg", "_1.jpg")
            else:
                cloth_filename = cloth_name

            cloth = imread(
                opj(self.drd, "cloth", cloth_filename),
                self.img_H,
                self.img_W
            )

            # cloth-mask
            if self.pair_key == "paired":
                mask_filename = cloth_name.replace("_0.jpg", "_1.png")
            else:
                mask_filename = cloth_name.replace(".jpg", ".png")  # 확장자만 변경

            cloth_mask = imread(
                opj(self.drd, "cloth-mask", mask_filename),
                self.img_H,
                self.img_W,
                is_mask=True,
                cloth_mask_check=True
            )
            
            gt_cloth_warped_mask = imread(
                opj(self.drd, "gt_cloth_warped_mask", self.im_names[idx]), 
                self.img_H, 
                self.img_W, 
                is_mask=True
            ) if not self.is_test else np.zeros_like(agn_mask)

            image = imread(opj(self.drd, "image", self.im_names[idx]), self.img_H, self.img_W)
            image_densepose = imread(opj(self.drd, "image-densepose", self.im_names[idx]), self.img_H, self.img_W)

            """dummy img"""
            wrp_name = "cloth-warp" if self.pair_key == "paired" else "unpaired-cloth-warp"
            wrp_mask_name = "cloth-warp-mask" if self.pair_key == "paired" else "unpaired-cloth-warp-mask"
            wrp_cloth = imread_for_albu(opj(self.drd, wrp_name, self.c_names[self.pair_key][idx])) if self.is_test else np.zeros_like(image)
            wrp_cloth_mask = imread_for_albu(opj(self.drd, wrp_mask_name, self.c_names[self.pair_key][idx]), is_mask=True) if self.is_test else np.zeros_like(image)

        else:
            agn = imread_for_albu(opj(self.drd, "agnostic-v3.2", self.im_names[idx]))
            agn_mask = imread_for_albu(opj(self.drd, "agnostic-mask", self.im_names[idx].replace(".jpg", "_mask.png")), is_mask=True)
            cloth_name = self.c_names[self.pair_key][idx]

            # cloth
            if self.pair_key == "paired":
                cloth_filename = cloth_name.replace("_0.jpg", "_1.jpg")
            else:
                cloth_filename = cloth_name
            cloth = imread_for_albu(opj(self.drd, "cloth", cloth_filename))

            # cloth-mask
            if self.pair_key == "paired":
                mask_filename = cloth_name.replace("_0.jpg", "_1.png")  # png로 확장자 바뀜
            else:
                mask_filename = cloth_name.replace(".jpg", ".png")  # unpaired도 png 사용
            cloth_mask = imread_for_albu(opj(self.drd, "cloth-mask", mask_filename), is_mask=True, cloth_mask_check=True)
            
            gt_cloth_warped_mask = imread_for_albu(
                opj(self.drd, "gt_cloth_warped_mask", self.im_names[idx]),
                is_mask=True
            ) if not self.is_test else np.zeros_like(agn_mask)
                
            image = imread_for_albu(opj(self.drd, "image", self.im_names[idx]))
            image_densepose = imread_for_albu(opj(self.drd, "image-densepose", self.im_names[idx]))

            if self.transform_size is not None:
                transformed = self.transform_size(
                    image=image, 
                    agn=agn, 
                    agn_mask=agn_mask, 
                    cloth=cloth, 
                    cloth_mask=cloth_mask, 
                    image_densepose=image_densepose,
                    gt_cloth_warped_mask=gt_cloth_warped_mask,
                )
                image=transformed["image"]
                agn=transformed["agn"]
                agn_mask=transformed["agn_mask"]
                image_densepose=transformed["image_densepose"]
                gt_cloth_warped_mask=transformed["gt_cloth_warped_mask"]

                cloth=transformed["cloth"]
                cloth_mask=transformed["cloth_mask"]

            """dummy img"""
            wrp_name = "cloth-warp" if self.pair_key == "paired" else "unpaired-cloth-warp"
            wrp_mask_name = "cloth-warp-mask" if self.pair_key == "paired" else "unpaired-cloth-warp-mask"
            # wrp_cloth = imread_for_albu(opj(self.drd, wrp_name, self.c_names[self.pair_key][idx])) if self.is_test else np.zeros_like(image)
            # wrp_cloth_mask = imread_for_albu(opj(self.drd, wrp_mask_name, self.c_names[self.pair_key][idx]), is_mask=True) if self.is_test else np.zeros_like(image)
            wrp_cloth = np.zeros_like(image)
            wrp_cloth_mask = np.zeros_like(image)

            if self.transform_crop_person is not None:
                transformed_image = self.transform_crop_person(
                    image=image,
                    agn=agn,
                    agn_mask=agn_mask,
                    image_densepose=image_densepose,
                    gt_cloth_warped_mask=gt_cloth_warped_mask,
                    wrp_cloth=wrp_cloth,
                    wrp_cloth_mask=wrp_cloth_mask,
                )

                image=transformed_image["image"]
                agn=transformed_image["agn"]
                agn_mask=transformed_image["agn_mask"]
                image_densepose=transformed_image["image_densepose"]
                gt_cloth_warped_mask=transformed_image["gt_cloth_warped_mask"]

                wrp_cloth=transformed_image["wrp_cloth"]
                wrp_cloth_mask=transformed_image["wrp_cloth_mask"]

            if self.transform_crop_cloth is not None:
                transformed_cloth = self.transform_crop_cloth(
                    image=cloth,
                    cloth_mask=cloth_mask
                )

                cloth=transformed_cloth["image"]
                cloth_mask=transformed_cloth["cloth_mask"]

            agn_mask = 255 - agn_mask
            if self.transform_color is not None:
                transformed = self.transform_color(
                    image=image, 
                    agn=agn, 
                    cloth=cloth,
                    wrp_cloth=wrp_cloth,
                )

                image=transformed["image"]
                agn=transformed["agn"]
                cloth=transformed["cloth"]
                
                wrp_cloth=transformed["wrp_cloth"]

                agn = agn * agn_mask[:,:,None].astype(np.float32)/255.0 + 128 * (1 - agn_mask[:,:,None].astype(np.float32)/255.0)
                
            agn = norm_for_albu(agn)
            agn_mask = norm_for_albu(agn_mask, is_mask=True)
            cloth = norm_for_albu(cloth)
            cloth_mask = norm_for_albu(cloth_mask, is_mask=True)
            image = norm_for_albu(image)
            image_densepose = norm_for_albu(image_densepose)
            gt_cloth_warped_mask = norm_for_albu(gt_cloth_warped_mask, is_mask=True)

            wrp_cloth = norm_for_albu(gt_cloth_warped_mask)
            wrp_cloth_mask = norm_for_albu(wrp_cloth_mask, is_mask=True)

        return dict(
            agn=agn,
            agn_mask=agn_mask,
            cloth=cloth,
            cloth_mask=cloth_mask,
            image=image,
            image_densepose=image_densepose,
            gt_cloth_warped_mask=gt_cloth_warped_mask,
            wrp_cloth=wrp_cloth,
            wrp_cloth_mask=wrp_cloth_mask,
            txt="",
            img_fn=img_fn,
            cloth_fn=cloth_fn,
        )