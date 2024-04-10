import numpy as np
import logging
import time
import torch
import torch.nn.functional as F
import os
from xmuda.data.utils.evaluate import Evaluator

from xmuda.data.utils.visualize_1 import draw_points_image_labels, draw_bird_eye_view


def viz(cfg,
        model_2d,
        model_3d,
        dataloader,
        val_metric_logger,
        pselab_path,
        output_dir):
    logger = logging.getLogger('xmuda.validate')
    logger.info('Validation')
    it = 0
    # evaluator
    class_names = dataloader.dataset.class_names
    evaluator_2d = Evaluator(class_names)
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_ensemble = Evaluator(class_names) if model_3d else None

    pselab_data_list = []

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end
            # copy data from cpu to gpu
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['seg_label'] = data_batch['seg_label'].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError

            # predict
            preds_2d = model_2d(data_batch)
            preds_3d = model_3d(data_batch) if model_3d else None

            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy()
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None

            # softmax average (ensembling)
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1) if model_3d else None
            pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy() if model_3d else None

            # get original point cloud from before voxelization
            seg_label = data_batch['orig_seg_label']
            points_idx = data_batch['orig_points_idx']
            # loop over batch
            left_idx = 0
            img_indices = data_batch['img_indices']
            for batch_ind in range(len(seg_label)):
                curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                assert np.all(curr_points_idx)

                curr_seg_label = seg_label[batch_ind]
                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx]
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if model_3d else None

                # evaluate
                evaluator_2d.update(pred_label_2d, curr_seg_label)
                if model_3d:
                    evaluator_3d.update(pred_label_3d, curr_seg_label)
                    evaluator_ensemble.update(pred_label_ensemble, curr_seg_label)
                # visual
                it = it + 1
                # igt = output_dir + 'viz/gt/'
                ipd = output_dir + 'viz/pred/'
                # if not os.path.exists(igt):  # 判断是否存在文件夹如果不存在则创建为文件夹
                #     os.makedirs(igt)
                if not os.path.exists(ipd):  # 判断是否存在文件夹如果不存在则创建为文件夹
                    os.makedirs(ipd)
                # igt = igt + str(it)
                ipd = ipd + str(it)
                # if it % 10 == 0:
                # draw_points_image_labels(data_batch['img'][batch_ind], img_indices[batch_ind], curr_seg_label, color_palette_type='NuScenes', point_size=3)
                # draw_points_image_labels((data_batch['img'][batch_ind].permute(1, 2, 0)).cpu(),
                #                             img_indices[batch_ind], curr_seg_label, igt,
                #                             color_palette_type='NuScenes', point_size=3)
                color_palette_type = ''
                if cfg.DATASET_TARGET.TYPE == 'NuScenesSCN':
                    color_palette_type = 'NuScenes'
                elif cfg.DATASET_TARGET.TYPE == 'SemanticKITTISCN':
                    color_palette_type = 'SemanticKITTI'
                draw_points_image_labels((data_batch['img'][batch_ind].permute(1, 2, 0)).cpu(),
                                         img_indices[batch_ind], pred_label_ensemble, ipd,
                                         color_palette_type=color_palette_type, point_size=3)
                # draw_bird_eye_view(coords)
                if pselab_path is not None:
                    assert np.all(pred_label_2d >= 0)
                    curr_probs_2d = probs_2d[left_idx:right_idx]
                    curr_probs_3d = probs_3d[left_idx:right_idx] if model_3d else None
                    pselab_data_list.append({
                        'probs_2d': curr_probs_2d[range(len(pred_label_2d)), pred_label_2d].cpu().numpy(),
                        'pseudo_label_2d': pred_label_2d.astype(np.uint8),
                        'probs_3d': curr_probs_3d[range(len(pred_label_3d)), pred_label_3d].cpu().numpy() if model_3d else None,
                        'pseudo_label_3d': pred_label_3d.astype(np.uint8) if model_3d else None
                    })

                left_idx = right_idx

            seg_loss_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch['seg_label'])
            seg_loss_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch['seg_label']) if model_3d else None
            val_metric_logger.update(seg_loss_2d=seg_loss_2d)
            if seg_loss_3d is not None:
                val_metric_logger.update(seg_loss_3d=seg_loss_3d)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        val_metric_logger.update(seg_iou_2d=evaluator_2d.overall_iou)
        if evaluator_3d is not None:
            val_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)
        eval_list = [('2D', evaluator_2d)]
        if model_3d:
            eval_list.extend([('3D', evaluator_3d), ('2D+3D', evaluator_ensemble)])
        for modality, evaluator in eval_list:
            logger.info('{} overall accuracy={:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU={:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))

        if pselab_path is not None:
            np.save(pselab_path, pselab_data_list)
            logger.info('Saved pseudo label data to {}'.format(pselab_path))


def viz_fusion(cfg,
                    model_2d,
                    model_3d,
                    model_fusion,
                    dataloader,
                    val_metric_logger,
                    pselab_path,
                    output_dir):
    logger = logging.getLogger('xmuda.validate')
    logger.info('Validation')
    it = 0
    # evaluator
    class_names = dataloader.dataset.class_names
    evaluator_2d = Evaluator(class_names)
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_ensemble = Evaluator(class_names) if model_3d else None

    pselab_data_list = []

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end
            # copy data from cpu to gpu
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['seg_label'] = data_batch['seg_label'].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError

            # predict
            preds_2d = model_2d(data_batch)
            preds_3d = model_3d(data_batch) if model_3d else None
            preds_fusion = model_fusion(preds_2d['feats_full'], preds_3d['feats'], data_batch['img_indices'])

            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy()
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None

            # softmax average (ensembling)
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1) if model_3d else None
            probs_fusion = F.softmax(preds_fusion, dim=1)
            pred_label_voxel_ensemble = probs_fusion.argmax(1).cpu().numpy() if model_3d else None
            # pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy() if model_3d else None

            # get original point cloud from before voxelization
            seg_label = data_batch['orig_seg_label']
            points_idx = data_batch['orig_points_idx']
            # loop over batch
            left_idx = 0
            img_indices = data_batch['img_indices']
            for batch_ind in range(len(seg_label)):
                curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                assert np.all(curr_points_idx)

                curr_seg_label = seg_label[batch_ind]
                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx]
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if model_3d else None

                # evaluate
                evaluator_2d.update(pred_label_2d, curr_seg_label)
                if model_3d:
                    evaluator_3d.update(pred_label_3d, curr_seg_label)
                    evaluator_ensemble.update(pred_label_ensemble, curr_seg_label)

                # visual
                it = it + 1
                # igt = './daysmgt/'
                # ipd = './daysmpd/'
                # igt = os.path.dirname(os.path.normpath(output_dir)) + '/gt/'
                # ipd = os.path.dirname(os.path.normpath(output_dir)) + '/pred/'
                igt = output_dir + 'viz/gt/'
                ipd = output_dir + 'viz/pred/'
                if not os.path.exists(igt):
                    os.makedirs(igt)
                if not os.path.exists(ipd):
                    os.makedirs(ipd)
                igt = igt + str(it)
                ipd = ipd + str(it)
                # if it % 10 == 0:
                # draw_points_image_labels(data_batch['img'][batch_ind], img_indices[batch_ind], curr_seg_label, color_palette_type='NuScenes', point_size=3)
                color_palette_type = ''
                if cfg.DATASET_TARGET.TYPE == 'NuScenesSCN':
                    color_palette_type = 'NuScenes'
                elif cfg.DATASET_TARGET.TYPE == 'SemanticKITTISCN':
                    color_palette_type = 'SemanticKITTI'

                draw_points_image_labels((data_batch['img'][batch_ind].permute(1, 2, 0)).cpu(),
                                         img_indices[batch_ind], curr_seg_label, igt,
                                         color_palette_type=color_palette_type, point_size=3)
                draw_points_image_labels((data_batch['img'][batch_ind].permute(1, 2, 0)).cpu(),
                                         img_indices[batch_ind], pred_label_ensemble, ipd,
                                         color_palette_type=color_palette_type, point_size=3)
                # draw_bird_eye_view(coords)

                if pselab_path is not None:
                    assert np.all(pred_label_2d >= 0)
                    curr_probs_2d = probs_2d[left_idx:right_idx]
                    curr_probs_3d = probs_3d[left_idx:right_idx] if model_3d else None
                    curr_probs_fusion = probs_fusion[left_idx:right_idx] if model_3d else None
                    pselab_data_list.append({
                        'probs_2d': curr_probs_2d[range(len(pred_label_2d)), pred_label_2d].cpu().numpy(),
                        'pseudo_label_2d': pred_label_2d.astype(np.uint8),
                        'probs_3d': curr_probs_3d[range(len(pred_label_3d)), pred_label_3d].cpu().numpy() if model_3d else None,
                        'pseudo_label_3d': pred_label_3d.astype(np.uint8) if model_3d else None,
                        'probs_fusion': curr_probs_fusion[
                            range(len(pred_label_ensemble)), pred_label_ensemble].cpu().numpy() if model_3d else None,
                        'pseudo_label_fusion': pred_label_ensemble.astype(np.uint8) if model_3d else None
                    })

                left_idx = right_idx

            seg_loss_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch['seg_label'])
            seg_loss_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch['seg_label']) if model_3d else None
            val_metric_logger.update(seg_loss_2d=seg_loss_2d)
            if seg_loss_3d is not None:
                val_metric_logger.update(seg_loss_3d=seg_loss_3d)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        val_metric_logger.update(seg_iou_2d=evaluator_2d.overall_iou)
        if evaluator_3d is not None:
            val_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)
        eval_list = [('2D', evaluator_2d)]
        if model_3d:
            eval_list.extend([('3D', evaluator_3d), ('2D+3D', evaluator_ensemble)])
        for modality, evaluator in eval_list:
            logger.info('{} overall accuracy={:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU={:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))

        if pselab_path is not None:
            np.save(pselab_path, pselab_data_list)
            logger.info('Saved pseudo label data to {}'.format(pselab_path))
