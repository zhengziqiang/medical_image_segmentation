#coding=utf-8
from PIL import Image
import numpy as np
from pylab import *
import os
import glob
import numpy as np
from hausdorff import hausdorff, weighted_hausdorff
'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)

        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_


def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(eval_segm)

    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def compute_identity(path):
    all_identity_output=[]
    all_identity_target = []
    for subdir in os.listdir(path):
        output_identity=np.zeros([1,240,240],np.uint8)
        target_identity=np.zeros([1,240,240],np.uint8)
        for i in range(155):
            file_name = os.path.join(path,subdir,"demo_" + str(i) + ".png")
            img = array(Image.open(file_name).convert("L"))
            output = img[:, 960:1200]
            target = img[:, 1200:1440]
            if target.min() > 50:
                target[:, :] = 0
            if output.min() > 50:
                output[:, :] = 0
            output_norm = np.where(output > 128, output, 0)
            target_norm = np.where(target > 128, target, 0)
            output_norm = np.where(output_norm < 128, output_norm, 255)
            target_norm = np.where(target_norm < 128, target_norm, 255)
            if i==0:
                output_norm=output_norm[np.newaxis,:,:]
                target_norm = target_norm[np.newaxis, :, :]
                output_identity=output_norm
                target_identity=target_norm
            else:
                tmp_output = output_norm[np.newaxis,:,:]
                tmp_target = target_norm[np.newaxis,:,:]
                output_identity=np.concatenate([output_identity,tmp_output],axis=0)
                target_identity = np.concatenate([target_identity, tmp_target], axis=0)
        all_identity_output.append(output_identity)
        all_identity_target.append(target_identity)
    return all_identity_output,all_identity_target


def mse(arr1,arr2):
    # arr1=array(img1)
    # arr2 = array(img2)
    arr1=arr1/255.0
    arr2 = arr2 / 255.0
    # mse_error_all+=(np.sqrt((np.sum((arr1-arr2)**2))/(256*256*3)))
    # mse_error_all += (np.sum((arr1 - arr2) ** 2) / (256 * 256 * 3))
    # rmse_error_all += np.sqrt(np.sum((arr1 - arr2) ** 2) / (256 * 256 * 3))

def evaluate_metric(output,target):
    list_acc = []
    list_iu = []
    list_fre = []
    list_mean_acc = []
    list_haus=[]
    list_dice=[]
    list_mse=[]
    list_rmse=[]
    cnt=0
    for i in range(len(output)):
        acc_all = 0.0
        mean_iu_all = 0.0
        fre_all = 0.0
        mean_acc_all = 0.0
        hausdorff_dis=0.0
        dice_all=0.0
        mse_all=0.0
        rmse_all=0.0
        data_output=output[i]
        data_target=target[i]
        for j in range(155):
            X=np.array(data_output[j,:,:]/255.0)
            Y=np.array(data_target[j,:,:]/255.0)
            mse_all += (((np.sum((X - Y) ** 2)) / (240 * 240)))
            rmse_all += np.sqrt(np.sum((X - Y) ** 2) / (240 * 240))
            if  np.max(X)>0.0 or np.max(Y)>0.0:
                dice_tmp = np.sum(X[Y == 1.0]) * 2.0 / (np.sum(X) + np.sum(Y))
                dice_all+=dice_tmp
                cnt+=1
            tmp_hausdorff=hausdorff(X, Y)
            hausdorff_dis+=tmp_hausdorff
            acc = pixel_accuracy(data_output[j,:,:], data_target[j,:,:])
            mean_iu = mean_IU(data_output[j,:,:], data_target[j,:,:])
            acc_all += acc
            mean_iu_all += mean_iu
            fre = frequency_weighted_IU(data_output[j,:,:], data_target[j,:,:])
            mean_acc = mean_accuracy(data_output[j,:,:], data_target[j,:,:])
            fre_all += fre
            mean_acc_all += mean_acc
        # print(acc_all / 155.0)
        # print(mean_iu_all / 155.0)
        # print(fre_all / 155.0)
        # print(mean_acc_all / 155.0)
        # print(hausdorff_dis / 155.0)
        list_acc.append(acc_all / 155.0)
        list_iu.append(mean_iu_all / 155.0)
        list_fre.append(fre_all / 155.0)
        list_mean_acc.append(mean_acc_all / 155.0)
        list_haus.append(hausdorff_dis/155.0)
        list_dice.append(dice_all/cnt)
        list_mse.append(mse_all/155.0)
        list_rmse.append(rmse_all/155.0)

    return list_acc,list_iu,list_fre,list_mean_acc,list_haus,list_dice,list_mse,list_rmse

if __name__ == '__main__':
    output,target=compute_identity("/media/vision/43c620be-e7c3-4af9-9cf6-c791ef2ed83e/zzq/medical_segmentation/p2p3d_t/gan_res/0019")
    acc,iu,fre,mean_acc,haus,dice,mse,rmse=evaluate_metric(output,target)
    acc_=np.sum(acc)/len(output)
    iu_=np.sum(iu) / len(output)
    fre_=np.sum(fre) / len(output)
    mean_acc_=np.sum(mean_acc) / len(output)
    haus_ = np.sum(haus) / len(haus)
    dice_=np.sum(dice) / len(dice)
    mse_ = np.sum(mse) / len(mse)
    rmse_ = np.sum(rmse) / len(rmse)
    print(acc_,iu_,fre_,mean_acc_,haus_,1.0-dice_,mse_,rmse_)

