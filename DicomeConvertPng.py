import SimpleITK as sitK
import numpy as np
import cv2
import os
def convert_from_dicom_to_png(img,low_window,high_window,save_path):
    lungwin = np.array([low_window*1.,high_window * 1.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])  #归一化
    newimg = (newimg*255).astype('uint8')  #扩展像素值到【0，255】
    cv2.imwrite(save_path,newimg)

if __name__ == '__main__':

    folder_path = "./NegtiveDicom/"
    jpg_folder_path = "./img"
    images_path = os.listdir(folder_path)
    for n, image in enumerate(images_path):

        dcm_image = os.path.join(folder_path, image)

        output_png_file = './img/' +'negtive_' + image + '.png'

        ds_array = sitK.ReadImage(dcm_image)
        img_array = sitK.GetArrayFromImage(ds_array)
        shape = img_array.shape
        img_array = np.reshape(img_array[:,:,:,0], (shape[1], shape[2]))
        high = np.max(img_array)
        low = np.min(img_array)
        convert_from_dicom_to_png(img_array, low, high, output_png_file)

