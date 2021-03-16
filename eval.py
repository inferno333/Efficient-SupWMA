
import numpy as np
import h5py
import time
import os
import pickle

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

from utils.logger import create_logger
from utils.model import PointNetCls
from utils.model_supcon import PointNet_SupCon, PointNet_Classifier
from utils.dataset import ORGDataset
from utils.metrics_plots import classify_report, calculate_entire_data_average_metric
from utils.funcs import makepath


def load_test_data(args, logger, num_fold):
    """load train and validation data"""
    # load feature and label data
    val_dataset = ORGDataset(
        root=args.input_path,
        logger=logger,
        num_fold=num_fold,
        k=args.k_fold,
        split='val')

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size,
        shuffle=True, num_workers=int(args.num_workers))

    val_data_size = len(val_dataset)
    logger.info('The validation data size is:{}'.format(val_data_size))
    num_classes = len(val_dataset.label_names)
    logger.info('The number of classes is:{}'.format(num_classes))

    # load label names
    label_names = val_dataset.obtain_label_names()
    label_names_h5 = h5py.File(os.path.join(args.out_path, 'label_names.h5'), 'w')
    label_names_h5['y_names'] = label_names
    logger.info('The label names are: {}'.format(str(label_names)))

    return val_loader, label_names, num_classes


def contrastive_two_stage_eval_net(stage1_params, encoder_params, args, stage1_net, stage2_encoder_net, stage2_classifer_net,
                                   test_data_loader, label_names, script_name, logger, log_res_path, device):
    """perform predition of two-stage model with contrastive loss"""
    logger.info('')
    logger.info('===================================')
    logger.info('')
    logger.info('{} Start multi-cluster prediction.'.format(script_name))

    output_prediction_report_path = os.path.join(log_res_path, 'entire_data_validation_results_best_{}.h5'.format(args.best_metric))
    # Load model
    start_time = time.time()
    with torch.no_grad():
        total_test_correct = 0
        test_labels_lst, test_predicted_lst, test_swm_labels_lst = [], [], []
        encoder_swm_features_array = None
        tot_swm_points = None
        for j, data in (enumerate(test_data_loader, 0)):
            points, labels = data
            points = points.transpose(2, 1)
            labels = labels[:, 0]
            points, labels = points.to(device), labels.to(device)
            stage1_net, stage2_encoder_net, stage2_classifer_net = \
                stage1_net.eval(), stage2_encoder_net.eval(), stage2_classifer_net.eval()

            # initialization
            tmp = torch.tensor(-1).to(device)
            pred_idx = tmp.repeat(points.shape[0])
            # stage 1
            stage1_pred = stage1_net(points)
            _, stage1_pred_idx = torch.max(stage1_pred, dim=1)
            stage1_swm_mask = torch.where(stage1_pred_idx < stage1_params['num_swm_stage1'])[0]
            stage1_other_mask = torch.where(stage1_pred_idx >= stage1_params['num_swm_stage1'])[0]
            pred_idx[stage1_other_mask] = torch.tensor(198).to(device)

            # stage 2
            if stage1_swm_mask.shape[0] != 0:
                swm_points = points[stage1_swm_mask, :, :]
                features = stage2_encoder_net.encoder(swm_points)
                stage2_pred = stage2_classifer_net(features)
                _, stage2_pred_idx = torch.max(stage2_pred, dim=1)
                pred_idx[stage1_swm_mask] = torch.where(stage2_pred_idx < 198, stage2_pred_idx,
                                                        torch.tensor(198).to(device))

            # entire data
            correct = pred_idx.eq(labels.data).cpu().sum()
            # for calculating test accuracy
            total_test_correct += correct.item()
            # for calculating test weighted and macro metrics
            labels = labels.cpu().detach().numpy().tolist()
            test_labels_lst.extend(labels)
            assert torch.sum(pred_idx == tmp) == 0
            pred_idx = pred_idx.cpu().detach().numpy().tolist()
            test_predicted_lst.extend(pred_idx)

    end_time = time.time()
    pred_time = end_time - start_time
    logger.info('The total time of prediction is:{} s'.format(round((pred_time), 4)))
    logger.info('The test sample size is: {}'.format(len(test_predicted_lst)))
    label_names_str = [str(label_name) for label_name in label_names]
    classify_report(test_labels_lst, test_predicted_lst, label_names_str, logger, output_prediction_report_path, '{}_test'.format(args.best_metric))

    return pred_time




def kfold_evaluate_two_stage_contrastive_model(stage1_params, encoder_params, args, device, script_name):