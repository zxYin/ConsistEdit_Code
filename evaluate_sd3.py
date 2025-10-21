import json
import argparse
import os
import numpy as np
from PIL import Image
import csv
from evaluation.matric_calculator import MetricsCalculator

def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array



def calculate_metric(metrics_calculator,metric, src_image, tgt_image, src_mask, tgt_mask,src_prompt,tgt_prompt):
    if metric=="contrast_sim":
        return metrics_calculator.calculate_contrast_sim(src_image, tgt_image)
    if metric=="tile_correlation":
        return metrics_calculator.calculate_tile_correlation(src_image, tgt_image)
    if metric=="bhattacharyya_sim":
        return metrics_calculator.calculate_bhattacharyya_sim(src_image, tgt_image)
    if metric=="gradient_sim":
        return metrics_calculator.calculate_gradient_sim(src_image, tgt_image)
    if metric=="psnr":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
    if metric=="lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    if metric=="mse":
        return metrics_calculator.calculate_mse(src_image, tgt_image, None, None)
    if metric=="ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
    if metric=="canny_ssim":
        return metrics_calculator.calculate_canny_ssim(src_image, tgt_image, None, None)
    if metric=="structure_distance":
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, None, None)
    if metric=="psnr_unedit_part":
        if src_mask is None or tgt_mask is None:
            return "nan"
        elif (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="lpips_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="mse_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="psnr_edit_part":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="ssim_unedit_part":
        if src_mask is None or tgt_mask is None:
            return "nan"
        elif (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="structure_distance_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="psnr_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="lpips_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="mse_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="ssim_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="structure_distance_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt,None)
    if metric=="clip_similarity_source_image_target_prompt":
        return metrics_calculator.calculate_clip_similarity(src_image, tgt_prompt,None)
    if metric=="clip_similarity_source_image_edit_part":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt, tgt_mask)
    if metric=="clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt,None)
    if metric=="clip_similarity_target_image_source_prompt":
        return metrics_calculator.calculate_clip_similarity(tgt_image, src_prompt,None)
    if metric=="clip_similarity_target_image_edit_part":
        if src_mask is None or tgt_mask is None:
            return "nan"
        elif tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt,tgt_mask)
    
all_tgt_image_folders={
    # results of comparing inversion
    # ---
    "ours":"results_output/ours/annotation_images",
    }


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_mapping_file', type=str, default="/comp_robot/yinzixin/MasaCtrl/PnPInversion/dataset/mapping_file.json")
    parser.add_argument('--metrics',  nargs = '+', type=str, default=[
                                                         "canny_ssim",
                                                         "structure_distance",
                                                         "psnr_unedit_part",
                                                         "ssim_unedit_part",
                                                         "clip_similarity_target_image",
                                                         "clip_similarity_target_image_edit_part",
                                                         ])
    parser.add_argument('--src_image_folder', type=str, default="ours/source_images")
    parser.add_argument('--mask_image_folder', type=str, default="ours/dilate_mask_images")
    parser.add_argument('--tgt_methods', nargs = '+', type=str, default=["ours"])
    parser.add_argument('--result_path', type=str, default="evaluation_result_ours.csv")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--edit_category_list',  nargs = '+', type=str, default=[
                                                                                "0",
                                                                                "1",
                                                                                "2",
                                                                                "3",
                                                                                "4",
                                                                                "5",
                                                                                "6",
                                                                                "7",
                                                                                "8",
                                                                                "9"
                                                                                ]) # the editing category that needed to run
    parser.add_argument('--evaluate_whole_table', action= "store_true") # rerun existing images

    args = parser.parse_args()
    
    annotation_mapping_file=args.annotation_mapping_file
    metrics=args.metrics
    src_image_folder=args.src_image_folder
    mask_image_folder=args.mask_image_folder
    tgt_methods=args.tgt_methods
    edit_category_list=args.edit_category_list
    evaluate_whole_table=args.evaluate_whole_table
    
    tgt_image_folders={}
    
    if evaluate_whole_table:
        for key in all_tgt_image_folders:
            if key[0] in tgt_methods:
                tgt_image_folders[key]=all_tgt_image_folders[key]
    else:
        for key in tgt_methods:
            tgt_image_folders[key]=all_tgt_image_folders[key]
    
    result_path=args.result_path
    
    metrics_calculator=MetricsCalculator(args.device)
    
    with open(result_path,'w',newline="") as f:
        csv_write = csv.writer(f)
        
        csv_head=[]
        for tgt_image_folder_key,_ in tgt_image_folders.items():
            for metric in metrics:
                csv_head.append(f"{tgt_image_folder_key}|{metric}")
        
        data_row = ["file_id"]+csv_head
        csv_write.writerow(data_row)

    with open(annotation_mapping_file,"r") as f:
        annotation_file=json.load(f)

    for key, item in annotation_file.items():
        if item["editing_type_id"] not in edit_category_list:
            continue
        print(f"evaluating image {key} ...")
        base_image_path=item["image_path"].replace(".jpg", ".png")
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        
        
        src_image_path=os.path.join(src_image_folder, base_image_path)
        mask_image_path=os.path.join(mask_image_folder, base_image_path)
        src_image = Image.open(src_image_path)

        if os.path.exists(mask_image_path):
            mask = np.array(Image.open(mask_image_path).resize(src_image.size))[:, :, np.newaxis].repeat([3],axis=2) / 255.0
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            mask = 1 - mask
        else:
            mask = None
        
        evaluation_result=[key]
        
        for tgt_image_folder_key,tgt_image_folder in tgt_image_folders.items():
            tgt_image_path=os.path.join(tgt_image_folder, base_image_path)
            print(f"evluating method: {tgt_image_folder_key}")
            print(tgt_image_path)

            if not os.path.exists(tgt_image_path):
                break
            
            tgt_image = Image.open(tgt_image_path)
            if tgt_image.size[0] != tgt_image.size[1]:
                # to evaluate editing
                tgt_image = tgt_image.crop((tgt_image.size[0]-512,tgt_image.size[1]-512,tgt_image.size[0],tgt_image.size[1])) 
            
            for metric in metrics:
                print(f"evluating metric: {metric}")
                evaluation_result.append(calculate_metric(metrics_calculator,metric,src_image, tgt_image, mask, mask, original_prompt, editing_prompt))
                        
        with open(result_path,'a+',newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(evaluation_result)
        
    # Calculate and write averages
    with open(result_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        
    # Calculate averages for each column (skip header and first column)
    averages = ['Average']
    for col in range(1, len(rows[0])):
        values = []
        for row in rows[1:]:  # Skip header
            try:
                value = float(row[col])
                if not np.isnan(value):  # Skip NaN values
                    values.append(value)
            except (ValueError, TypeError):
                continue
        if values:
            avg = sum(values) / len(values)
            averages.append(f"{avg:.4f}")
        else:
            averages.append("nan")
            
    # Write averages as the last row
    with open(result_path, 'a', newline="") as f:
        csv_write = csv.writer(f)
        csv_write.writerow(averages)
        
        