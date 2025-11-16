import copy
import os
from abc import ABCMeta, abstractmethod

import json_tricks as json
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_epe, keypoint_nme,
                                                  keypoint_pck_accuracy)
from mmpose.datasets import DATASETS
from mmpose.datasets.pipelines import Compose
from torch.utils.data import Dataset
import random


@DATASETS.register_module()
class TestBaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=True,
                 PCK_threshold_list=[0.05, 0.1, 0.15, 0.2, 0.25]):
        self.image_info = {}
        self.ann_info = {}

        self.annotations_path = ann_file
        if not img_prefix.endswith('/'):
            img_prefix = img_prefix + '/'
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.PCK_threshold_list = PCK_threshold_list

        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']

        self.ann_info['flip_pairs'] = None

        self.ann_info['inference_channel'] = data_cfg['inference_channel']
        self.ann_info['num_output_channels'] = data_cfg['num_output_channels']
        self.ann_info['dataset_channel'] = data_cfg['dataset_channel']

        self.db = []
        self.num_shots = 1
        self.paired_samples = []
        self.pipeline = Compose(self.pipeline)

    @abstractmethod
    def _get_db(self):
        """Load dataset."""
        raise NotImplementedError

    @abstractmethod
    def _select_kpt(self, obj, kpt_id):
        """Select kpt."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """Evaluate keypoint results."""
        raise NotImplementedError

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _report_metric(self,
                       res_file,
                       metrics):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'PCKh', 'AUC', 'EPE'.
            pck_thr (float): PCK threshold, default as 0.2.
            pckh_thr (float): PCKh threshold, default as 0.7.
            auc_nor (float): AUC normalization factor, default as 30 pixel.

        Returns:
            List: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.paired_samples)

        outputs = []
        gts = []
        masks = []
        threshold_bbox = []
        threshold_head_box = []

        for pred, pair in zip(preds, self.paired_samples):
            item = self.db[pair[-1]]
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])

            mask_query = ((np.array(item['joints_3d_visible'])[:, 0]) > 0)
            mask_sample = ((np.array(self.db[pair[0]]['joints_3d_visible'])[:, 0]) > 0)
            for id_s in pair[:-1]:
                mask_sample = np.bitwise_and(mask_sample, ((np.array(self.db[id_s]['joints_3d_visible'])[:, 0]) > 0))
            masks.append(np.bitwise_and(mask_query, mask_sample))

            if 'PCK' in metrics or 'NME' in metrics or 'AUC' in metrics:
                bbox = np.array(item['bbox'])
                bbox_thr = np.max(bbox[2:])
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))
            if 'PCKh' in metrics:
                head_box_thr = item['head_size']
                threshold_head_box.append(
                    np.array([head_box_thr, head_box_thr]))

        if 'PCK' in metrics:
            pck_results = dict()
            for pck_thr in self.PCK_threshold_list:
                pck_results[pck_thr] = []

            for (output, gt, mask, thr_bbox) in zip(outputs, gts, masks, threshold_bbox):
                for pck_thr in self.PCK_threshold_list:
                    _, pck, _ = keypoint_pck_accuracy(np.expand_dims(output, 0), np.expand_dims(gt, 0),
                                                      np.expand_dims(mask, 0), pck_thr, np.expand_dims(thr_bbox, 0))
                    pck_results[pck_thr].append(pck)

            mPCK = 0
            for pck_thr in self.PCK_threshold_list:
                info_str.append(['PCK@' + str(pck_thr), np.mean(pck_results[pck_thr])])
                mPCK += np.mean(pck_results[pck_thr])
            info_str.append(['mPCK', mPCK / len(self.PCK_threshold_list)])

        if 'NME' in metrics:
            nme_results = []
            for (output, gt, mask, thr_bbox) in zip(outputs, gts, masks, threshold_bbox):
                nme = keypoint_nme(np.expand_dims(output, 0), np.expand_dims(gt, 0), np.expand_dims(mask, 0),
                                   np.expand_dims(thr_bbox, 0))
                nme_results.append(nme)
            info_str.append(['NME', np.mean(nme_results)])

        if 'AUC' in metrics:
            auc_results = []
            for (output, gt, mask, thr_bbox) in zip(outputs, gts, masks, threshold_bbox):
                auc = keypoint_auc(np.expand_dims(output, 0), np.expand_dims(gt, 0), np.expand_dims(mask, 0),
                                   thr_bbox[0])
                auc_results.append(auc)
            info_str.append(['AUC', np.mean(auc_results)])

        if 'EPE' in metrics:
            epe_results = []
            for (output, gt, mask) in zip(outputs, gts, masks):
                epe = keypoint_epe(np.expand_dims(output, 0), np.expand_dims(gt, 0), np.expand_dims(mask, 0))
                epe_results.append(epe)
            info_str.append(['EPE', np.mean(epe_results)])
        return info_str

    def _report_supercategory_metric(self, res_file, metrics):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'PCKh', 'AUC', 'EPE'.
            pck_thr (float): PCK threshold, default as 0.2.
            pckh_thr (float): PCKh threshold, default as 0.7.
            auc_nor (float): AUC normalization factor, default as 30 pixel.

        Returns:
            List: Evaluation results for evaluation metric.
        """

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.paired_samples)

        supercategory2predIndex = dict()
        categoryId2supercategory = dict()
        for catInfo in self.coco.cats.values():
            if catInfo['name'].endswith('_body') and len(catInfo['keypoints']) == 17:
                supercategoryName = "animal_body"
            else:
                supercategoryName = catInfo['supercategory']
            supercategory2predIndex[supercategoryName] = []
            categoryId2supercategory[catInfo['id']] = supercategoryName

        for pred_index, pred in enumerate(preds):
            predImgId = pred['image_id']
            categoryId = self.coco.loadAnns(self.coco.getAnnIds(imgIds=predImgId))[0]['category_id']
            supercategoryName = categoryId2supercategory[categoryId]
            supercategory2predIndex[supercategoryName].append(pred_index)

        all_info_str = []
        all_supercategory_pck = dict()
        for supercategoryName in supercategory2predIndex.keys():
            outputs = []
            gts = []
            masks = []
            threshold_bbox = []
            threshold_head_box = []
            info_str = []

            relatedPredIndex = supercategory2predIndex[supercategoryName]
            # 根据relatedPredIndex获取相应索引的preds和self.paired_samples
            relatedPreds = [preds[i] for i in relatedPredIndex]
            relatedPairedSamples = np.array([self.paired_samples[i] for i in relatedPredIndex])
            for pred, pair in zip(relatedPreds, relatedPairedSamples):
                item = self.db[pair[-1]]
                outputs.append(np.array(pred['keypoints'])[:, :-1])
                gts.append(np.array(item['joints_3d'])[:, :-1])

                mask_query = ((np.array(item['joints_3d_visible'])[:, 0]) > 0)
                mask_sample = ((np.array(self.db[pair[0]]['joints_3d_visible'])[:, 0]) > 0)
                for id_s in pair[:-1]:
                    mask_sample = np.bitwise_and(mask_sample, ((np.array(self.db[id_s]['joints_3d_visible'])[:, 0]) > 0))
                masks.append(np.bitwise_and(mask_query, mask_sample))

                if 'PCK' in metrics or 'NME' in metrics or 'AUC' in metrics:
                    bbox = np.array(item['bbox'])
                    bbox_thr = np.max(bbox[2:])
                    threshold_bbox.append(np.array([bbox_thr, bbox_thr]))
                if 'PCKh' in metrics:
                    head_box_thr = item['head_size']
                    threshold_head_box.append(
                        np.array([head_box_thr, head_box_thr]))

            if 'PCK' in metrics:
                pck_results = dict()
                for pck_thr in self.PCK_threshold_list:
                    pck_results[pck_thr] = []

                for (output, gt, mask, thr_bbox) in zip(outputs, gts, masks, threshold_bbox):
                    for pck_thr in self.PCK_threshold_list:
                        _, pck, _ = keypoint_pck_accuracy(np.expand_dims(output, 0), np.expand_dims(gt, 0),
                                                          np.expand_dims(mask, 0), pck_thr, np.expand_dims(thr_bbox, 0))
                        pck_results[pck_thr].append(pck)

                # pck_thr = 0.2
                # info_str.append(['{} PCK@{} mean'.format(supercategoryName, pck_thr), np.mean(pck_results[pck_thr])])
                # info_str.append(['{} PCK@{} std'.format(supercategoryName, pck_thr), np.std(pck_results[pck_thr])])

                all_supercategory_pck[supercategoryName] = dict()
                mPCK = 0
                for pck_thr in self.PCK_threshold_list:
                    info_str.append(['{} PCK@{} mean'.format(supercategoryName, pck_thr), np.mean(pck_results[pck_thr])])
                    info_str.append(['{} PCK@{} std'.format(supercategoryName, pck_thr), np.std(pck_results[pck_thr])])
                    mPCK += np.mean(pck_results[pck_thr])
                    all_supercategory_pck[supercategoryName][pck_thr] = {'mean': np.mean(pck_results[pck_thr]),
                                                                         'std': np.std(pck_results[pck_thr])}
                info_str.append(['{} mPCK'.format(supercategoryName), mPCK / len(self.PCK_threshold_list)])
            all_info_str.extend(info_str)

        work_dir = os.path.dirname(res_file)
        self._draw_supCat_pck(all_supercategory_pck, work_dir)

        return all_info_str

    def _report_category_metric(self, res_file, metrics):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'PCKh', 'AUC', 'EPE'.
            pck_thr (float): PCK threshold, default as 0.2.
            pckh_thr (float): PCKh threshold, default as 0.7.
            auc_nor (float): AUC normalization factor, default as 30 pixel.

        Returns:
            List: Evaluation results for evaluation metric.
        """

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.paired_samples)

        category2predIndex = dict()
        for catInfo in self.coco.cats.values():
            categoryName = catInfo['name']
            category2predIndex[categoryName] = []

        for pred_index, pred in enumerate(preds):
            predImgId = pred['image_id']
            categoryId = self.coco.loadAnns(self.coco.getAnnIds(imgIds=predImgId))[0]['category_id']
            categoryName = self.coco.loadCats(ids=categoryId)[0]['name']
            category2predIndex[categoryName].append(pred_index)

        all_info_str = []
        all_category_pck = dict()
        for categoryName in category2predIndex.keys():
            outputs = []
            gts = []
            masks = []
            threshold_bbox = []
            threshold_head_box = []
            info_str = []

            relatedPredIndex = category2predIndex[categoryName]
            # 根据relatedPredIndex获取相应索引的preds和self.paired_samples
            relatedPreds = [preds[i] for i in relatedPredIndex]
            relatedPairedSamples = np.array([self.paired_samples[i] for i in relatedPredIndex])
            for pred, pair in zip(relatedPreds, relatedPairedSamples):
                item = self.db[pair[-1]]
                outputs.append(np.array(pred['keypoints'])[:, :-1])
                gts.append(np.array(item['joints_3d'])[:, :-1])

                mask_query = ((np.array(item['joints_3d_visible'])[:, 0]) > 0)
                mask_sample = ((np.array(self.db[pair[0]]['joints_3d_visible'])[:, 0]) > 0)
                for id_s in pair[:-1]:
                    mask_sample = np.bitwise_and(mask_sample, ((np.array(self.db[id_s]['joints_3d_visible'])[:, 0]) > 0))
                masks.append(np.bitwise_and(mask_query, mask_sample))

                if 'PCK' in metrics or 'NME' in metrics or 'AUC' in metrics:
                    bbox = np.array(item['bbox'])
                    bbox_thr = np.max(bbox[2:])
                    threshold_bbox.append(np.array([bbox_thr, bbox_thr]))
                if 'PCKh' in metrics:
                    head_box_thr = item['head_size']
                    threshold_head_box.append(
                        np.array([head_box_thr, head_box_thr]))

            if 'PCK' in metrics:
                pck_results = dict()
                for pck_thr in self.PCK_threshold_list:
                    pck_results[pck_thr] = []

                for (output, gt, mask, thr_bbox) in zip(outputs, gts, masks, threshold_bbox):
                    for pck_thr in self.PCK_threshold_list:
                        _, pck, _ = keypoint_pck_accuracy(np.expand_dims(output, 0), np.expand_dims(gt, 0),
                                                          np.expand_dims(mask, 0), pck_thr, np.expand_dims(thr_bbox, 0))
                        pck_results[pck_thr].append(pck)

                # pck_thr = 0.2
                # info_str.append(['{} PCK@{} mean'.format(supercategoryName, pck_thr), np.mean(pck_results[pck_thr])])
                # info_str.append(['{} PCK@{} std'.format(supercategoryName, pck_thr), np.std(pck_results[pck_thr])])

                all_category_pck[categoryName] = dict()
                mPCK = 0
                for pck_thr in self.PCK_threshold_list:
                    info_str.append(['{} PCK@{} mean'.format(categoryName, pck_thr), np.mean(pck_results[pck_thr])])
                    info_str.append(['{} PCK@{} std'.format(categoryName, pck_thr), np.std(pck_results[pck_thr])])
                    mPCK += np.mean(pck_results[pck_thr])
                    all_category_pck[categoryName][pck_thr] = {'mean': np.mean(pck_results[pck_thr]),
                                                               'std': np.std(pck_results[pck_thr])}
                info_str.append(['{} mPCK'.format(categoryName), mPCK / len(self.PCK_threshold_list)])
            all_info_str.extend(info_str)

        work_dir = os.path.dirname(res_file)
        self._draw_supCat_pck(all_category_pck, work_dir)

        return all_info_str

    def _draw_supCat_pck(self, all_supercategory_pck, work_dir):
        from matplotlib import pyplot as plt
        supCatNames = list(all_supercategory_pck.keys())
        for pck_thr in self.PCK_threshold_list:
            pck_thr_mean = []
            pck_thr_std = []
            for supCatName in supCatNames:
                pck_thr_mean.append(all_supercategory_pck[supCatName][pck_thr]['mean'])
                pck_thr_std.append(all_supercategory_pck[supCatName][pck_thr]['std'])
            # 使用plt绘制折线图并保存至work_dir，横坐标为supCatNames，纵坐标为pck_thr_mean和pck_thr_std
            plt.figure()
            plt.title('PCK@' + str(pck_thr))
            plt.ylim(0, 1.1)
            plt.plot(supCatNames, pck_thr_mean, label='mean')
            plt.plot(supCatNames, pck_thr_std, label='std')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(work_dir, "test_PCK@{}.png".format(pck_thr)))
            plt.close()
            print('save PCK_' + str(pck_thr) + ' to ' + work_dir)

    def _merge_obj(self, Xs_list, Xq, idx):
        """ merge Xs_list and Xq.

        :param Xs_list: N-shot samples X
        :param Xq: query X
        :param idx: id of paired_samples
        :return: Xall
        """
        Xall = dict()
        Xall['img_s'] = [Xs['img'] for Xs in Xs_list]
        Xall['target_s'] = [Xs['target'] for Xs in Xs_list]
        Xall['target_weight_s'] = [Xs['target_weight'] for Xs in Xs_list]
        xs_img_metas = [Xs['img_metas'].data for Xs in Xs_list]

        Xall['img_q'] = Xq['img']
        Xall['target_q'] = Xq['target']
        Xall['target_weight_q'] = Xq['target_weight']
        xq_img_metas = Xq['img_metas'].data

        img_metas = dict()
        for key in xq_img_metas.keys():
            img_metas['sample_' + key] = [xs_img_meta[key] for xs_img_meta in xs_img_metas]
            img_metas['query_' + key] = xq_img_metas[key]
        img_metas['bbox_id'] = idx

        Xall['img_metas'] = DC(img_metas, cpu_only=True)

        return Xall

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.paired_samples)

    def __getitem__(self, idx):
        """Get the sample given index."""

        pair_ids = self.paired_samples[idx]  # [supported id * shots, query id]
        assert len(pair_ids) == self.num_shots + 1
        sample_id_list = pair_ids[:self.num_shots]
        query_id = pair_ids[-1]

        sample_obj_list = []
        for sample_id in sample_id_list:
            sample_obj = copy.deepcopy(self.db[sample_id])
            sample_obj['ann_info'] = copy.deepcopy(self.ann_info)
            sample_obj_list.append(sample_obj)

        query_obj = copy.deepcopy(self.db[query_id])
        query_obj['ann_info'] = copy.deepcopy(self.ann_info)

        Xs_list = []
        for sample_obj in sample_obj_list:
            Xs = self.pipeline(sample_obj)  # dict with ['img', 'target', 'target_weight', 'img_metas'],
            Xs['img_metas'].data['point_descriptions'] = self.cats_points_descriptions[
                Xs['img_metas'].data['category_id']]
            Xs['img_metas'].data['category_name'] = sample_obj['category_name']

            # train with random blank text prompt
            # joint_descriptions = self.cats_points_descriptions[Xs['img_metas'].data['category_id']]
            # num_elements = len(joint_descriptions)
            # empty_ratio = random.uniform(0.0, 0.3)
            # num_to_empty = int(num_elements * empty_ratio)
            # random_indices = random.sample(range(num_elements), num_to_empty)
            # for index in random_indices:
            #     joint_descriptions[index] = ' '
            # Xs['img_metas'].data['point_descriptions'] = joint_descriptions

            Xs_list.append(Xs)  # Xs['target'] is of shape [100, map_h, map_w]
        Xq = self.pipeline(query_obj)
        Xq['img_metas'].data['point_descriptions'] = self.cats_points_descriptions[Xq['img_metas'].data['category_id']]
        Xq['img_metas'].data['category_name'] = query_obj['category_name']

        Xall = self._merge_obj(Xs_list, Xq, idx)
        Xall['skeleton'] = self.db[query_id]['skeleton']

        return Xall

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts
