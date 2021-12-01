import os
import ipdb

import torch
from torch.nn import functional as F
import numpy as np
import pandas

from metrics import eval_func, eval_func_mp
from loss.triplet_loss import normalize, euclidean_dist
from functools import reduce

from metrics.rerank import re_ranking


def clck_dist(feat1, feat2, vis_score1, vis_score2, split=0):
    """
    计算vpm论文中的clck距离

    :param torch.Tensor feat1: [B1, C, 3]
    :param torch.Tensor feat2: [B2, C, 3]
    :param torch.Tensor vis_score: [B, 3]
    :rtype torch.Tensor
    :return: clck distance. [B1, B2]
    """
    vis_score1 = torch.exp(vis_score1)
    vis_score2 = torch.exp(vis_score2)

    B, C, N = feat1.shape
    dist_mat = 0
    ckcl = 0
    for i in range(N):
        parse_feat1 = feat1[:, :, i]
        parse_feat2 = feat2[:, :, i]
        ckcl_ = torch.mm(vis_score1[:, i].view(-1, 1),
                         vis_score2[:, i].view(1, -1))  # [N, N]
        ckcl += ckcl_
        dist_mat += euclidean_dist(parse_feat1,
                                   parse_feat2, split=split).sqrt() * ckcl_

    return dist_mat / ckcl


def calculate_feature_distance(feat1, feat2):
    m, n, color_num = feat1.shape[0], feat2.shape[0], feat1.shape[1]
    color_distmat = torch.pow(feat1, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(feat2, 2).sum(dim=1,
                                                                                                        keepdim=True).expand(
        n, m).t()
    color_distmat.addmm_(feat1, feat2.t(), beta=1, alpha=-2)
    return color_distmat.detach().cpu().numpy()


class Clck_R1_mAP:
    def __init__(self, num_query, *, max_rank=50, feat_norm=True, output_path='', rerank=False, remove_junk=True,
                 alpha=1, beta=0.5, lambda1=0, lambda2=0):
        """
        计算VPM中的可见性距离并计算性能

        :param num_query:
        :param max_rank:
        :param feat_norm:
        :param output_path:
        :param rerank:
        :param remove_junk:
        :param lambda_: distmat = global_dist + lambda_ * local_dist, default 0.5
        """
        super(Clck_R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.output_path = output_path
        self.rerank = rerank
        self.remove_junk = remove_junk
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.reset()

    def reset(self):
        self.global_feats = []
        self.local_feats = []
        self.vis_scores = []
        self.pids = []
        self.camids = []
        self.paths = []
        self.color_feature = []
        self.vehicle_type_feature = []

    def update(self, output):
        global_feat, local_feat, vis_score, pid, camid, paths, color_feature, vehicle_type_feature = output
        self.global_feats += global_feat
        self.local_feats += local_feat
        self.vis_scores += vis_score
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.paths += paths
        self.color_feature += color_feature
        self.vehicle_type_feature += vehicle_type_feature

    def save(self, path):
        output_dict = {
            "global_feats": self.global_feats,
            "local_feats": self.local_feats,
            "vis_scores": self.vis_scores,
            "pids": self.pids,
            "camids": self.camids,
            "paths": self.paths,
            "color_feature": self.color_feature,
            "vehicle_type_feature": self.vehicle_type_feature
        }
        torch.save(output_dict, path)

    def load(self, path):
        dict = torch.load(path)
        self.global_feats = dict["global_feats"]
        self.local_feats = dict["local_feats"]
        self.vis_scores = dict["vis_scores"]
        self.pids = dict["pids"]
        self.camids = dict["camids"]
        self.paths = dict["paths"]
        self.color_feature = dict["color_feature"]
        self.vehicle_type_feature = dict["vehicle_type_feature"]

    def resplit_for_vehicleid(self):
        """每个ID随机选择一辆车组成gallery，剩下的为query。
        """

        # 采样
        indexes = range(len(self.pids))
        df = pandas.DataFrame(dict(index=indexes, pid=self.pids))
        query_idxs = []
        gallery_idxs = []
        for idx, group in df.groupby('pid'):
            gallery = group.sample(1)['index'].iloc[0]
            gallery_idxs.append(gallery)
            for index in group.index:
                if index != gallery:
                    query_idxs.append(index)
        re_idxs = query_idxs + gallery_idxs

        self.num_query = len(query_idxs)
        # 重排序
        self.global_feats = [self.global_feats[i] for i in re_idxs]
        self.local_feats = [self.local_feats[i] for i in re_idxs]
        self.vis_scores = [self.vis_scores[i] for i in re_idxs]
        self.pids = [self.pids[i] for i in re_idxs]
        self.camids = [self.camids[i] for i in re_idxs]
        self.paths = [self.paths[i] for i in re_idxs]

    def compute(self, split=0, **kwargs):
        global_feats = torch.stack(self.global_feats, dim=0)
        local_feats = torch.stack(self.local_feats, dim=0)
        vis_scores = torch.stack(self.vis_scores)
        color_feature = torch.stack(self.color_feature, dim=0)
        vehicle_type_feature = torch.stack(self.vehicle_type_feature, dim=0)
        if self.feat_norm:
            print("The feature is normalized")
            global_feats = F.normalize(global_feats, dim=1, p=2)
            local_feats = F.normalize(local_feats, dim=1, p=2)
            color_feature = F.normalize(color_feature, dim=1, p=2)
            vehicle_type_feature = F.normalize(vehicle_type_feature, dim=1, p=2)
            vis_scores = F.normalize(vis_scores, dim=1, p=2)
        # 全局距离
        print('Calculate distance matrixs...')
        # query
        qf = global_feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallerye
        gf = global_feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        qf = qf
        m, n = qf.shape[0], gf.shape[0]

        if self.rerank:
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        else:
            # qf: M, F
            # gf: N, F
            if split == 0:
                distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                          torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
            else:
                distmat = gf.new(m, n)
                start = 0
                while start < n:
                    end = start + split if (start + split) < n else n
                    num = end - start

                    sub_distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, num) + \
                                  torch.pow(gf[start:end], 2).sum(
                                      dim=1, keepdim=True).expand(num, m).t()
                    # sub_distmat.addmm_(1, -2, qf, gf[start:end].t())
                    sub_distmat.addmm_(qf, gf[start:end].t(), beta=1, alpha=-2)
                    distmat[:, start:end] = sub_distmat

                    start += num

            distmat = distmat.detach().numpy()

        # 局部距离
        print('Calculate local distances...')
        local_distmat = clck_dist(local_feats[:self.num_query], local_feats[self.num_query:],
                                  vis_scores[:self.num_query], vis_scores[self.num_query:], split=split)

        local_feats = local_feats
        local_distmat = local_distmat.detach().cpu().numpy()

        # 颜色距离
        print('Calculate color distances.....')
        color_distmat = calculate_feature_distance(color_feature[:self.num_query], color_feature[self.num_query:])
        # 车型距离
        vehicle_type_distmat = calculate_feature_distance(vehicle_type_feature[:self.num_query],
                                                          vehicle_type_feature[self.num_query:])
        final_dismat = distmat * self.alpha + local_distmat * self.beta + color_distmat * self.lambda1 + vehicle_type_distmat * self.lambda2
        if self.output_path:
            print('Saving results...')
            outputs = {
                "global_feats": global_feats,
                "vis_scores": vis_scores,
                "local_feats": local_feats,
                "color_feature": self.color_feature,
                "vehicle_type_feature": self.vehicle_type_feature,
                "pids": self.pids,
                "camids": self.camids,
                "paths": self.paths,
                "num_query": self.num_query,
                "distmat": distmat,
                "local_distmat": local_distmat,
                "color_distmat": color_distmat,
                "vehicle_type_distmat": vehicle_type_distmat,
                "final_dismat": final_dismat
            }
            torch.save(outputs, os.path.join(self.output_path,
                                             'test_output.pkl'), pickle_protocol=4)

        if not kwargs['infer_flag']:
            print('Eval...')
            cmc, mAP, all_AP = eval_func_mp(final_dismat, q_pids, g_pids, q_camids, g_camids,
                                            remove_junk=self.remove_junk)

            return {
                "cmc": cmc,
                "mAP": mAP,
                "distmat": distmat,
                "all_AP": all_AP
            }
        else:
            return outputs

    def sort(self):
        pass
