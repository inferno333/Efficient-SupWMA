
import numpy as np
import h5py
import os
import sys
import copy
import torch
import matplotlib.ticker as mtick
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN


sys.path.append('..')
from utils.logger import create_logger
from utils.funcs import round_decimal_percentage, round_decimal


def calculate_prec_recall_f1(labels_lst, predicted_lst):
    # Beta: The strength of recall versus precision in the F-score. beta == 1.0 means recall and precision are equally important, that is F1-score
    mac_precision, mac_recall, mac_f1, _ = precision_recall_fscore_support(y_true=labels_lst, y_pred=predicted_lst, beta=1.0, average='macro')
    return mac_precision, mac_recall, mac_f1


def classify_report(labels_lst, predicted_lst, label_names, logger, out_path, metric_name):
    """Generate classification performance report"""
    cls_report = classification_report(y_true=labels_lst, y_pred=predicted_lst, digits=5, target_names=label_names)
    logger.info('=' * 55)
    logger.info('Best {} classification report:\n{}'.format(metric_name, cls_report))
    logger.info('=' * 55)
    logger.info('\n')

    if 'test' in metric_name:
        test_res = h5py.File(out_path, "w")
        test_res['val_predictions'] = predicted_lst
        test_res['val_labels'] = labels_lst
        test_res['label_names'] = label_names
        test_res['classification_report'] = cls_report
    else:
        val_res = h5py.File(os.path.join(out_path, 'entire_data_validation_results_best_{}.h5'.format(metric_name)), "w")
        val_res['val_predictions'] = predicted_lst
        val_res['val_labels'] = labels_lst
        val_res['label_names'] = label_names
        val_res['classification_report'] = cls_report


def per_class_metric(labels_lst, predicted_lst, label_names, val_data_size, logger, out_path, metric_name):
    """Analysis for each class metric and its metric"""
    cls_report_dict = classification_report(y_true=labels_lst, y_pred=predicted_lst, digits=5,
                                            target_names=label_names, output_dict=True)
    ratio_lst = []
    precision_lst = []
    recall_lst = []
    f1_lst = []