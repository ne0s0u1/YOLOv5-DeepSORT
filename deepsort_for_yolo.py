# deepsort_for_yolo.py
# (Version 2.0 - Final, Corrected)

import numpy as np
import torch
import torch.nn as nn
import torchvision
import cv2
import scipy
from scipy.optimize import linear_sum_assignment

# ##################################################################
# ALL REQUIRED CLASSES AND FUNCTIONS ARE NOW DEFINED AT THE TOP LEVEL
# ##################################################################

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking things in a 2D space.
    """
    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        self.chi2inv95 = {
            1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877,
            5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919}

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        d = measurements - mean
        cholesky_factor = np.linalg.cholesky(covariance)
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

def iou(bbox, candidates):
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]
    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)
    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)

def iou_cost(tracks, dets, track_indices, detection_indices):
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row_idx, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        for col_idx, det_idx in enumerate(detection_indices):
            det = dets[det_idx]
            cost_matrix[row_idx, col_idx] = 1 - iou(track.to_tlwh(), np.array([det.tlwh]))[0]
    return cost_matrix

class NearestNeighborDistanceMetric(object):
    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "cosine":
            self._metric = self._nn_cosine_distance
        else:
            raise ValueError("Invalid metric")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix

    def _nn_cosine_distance(self, x, y):
        distances = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                distances[i, j] = 1. - np.dot(x[i], y[j].T)
        return distances.min(axis=0)

# --- Re-ID Model Definition ---
class Net(nn.Module):
    def __init__(self, num_classes=751, resnet=None):
        super(Net, self).__init__()
        if resnet is None:
            resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.base = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net() # <-- No 'reid=True' here
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=torch.device(self.device))['net_dict']
        self.net.load_state_dict(state_dict, strict=False) # <-- Use strict=False
        self.net.to(self.device)
        self.net.eval()
        self.size = (128, 64)
        self.norm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        im_batch = []
        for im in im_crops:
            # Convert BGR to RGB
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, self.size)
            im_tensor = self.norm(im).unsqueeze(0)
            im_batch.append(im_tensor)
        im_batch = torch.cat(im_batch, dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        if not im_crops:
            return np.array([])
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()

# --- DeepSORT Core Classes ---
class Detection(object):
    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

class Track:
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = 'tentative'
        self.features = []
        if feature is not None:
            self.features.append(feature)
        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == 'tentative' and self.hits >= self._n_init:
            self.state = 'confirmed'

    def mark_missed(self):
        if self.state == 'tentative':
            self.state = 'deleted'
        elif self.time_since_update > self._max_age:
            self.state = 'deleted'

    def is_confirmed(self):
        return self.state == 'confirmed'
    def is_deleted(self):
        return self.state == 'deleted'

def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(float)
    pick = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > max_bbox_overlap)[0])))
    return pick

class Tracker:
    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        
        # Gated metric matching for confirmed tracks
        matches_a, unmatched_tracks_a, unmatched_detections = self._associate(
            detections, confirmed_tracks, self.metric.matching_threshold, "cosine")
        
        # IOU matching for tentative and unmatched confirmed tracks
        iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = self._associate(
            detections, iou_track_candidates, self.max_iou_distance, "iou", unmatched_detections)
        
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection.feature))
        self._next_id += 1

    def _associate(self, detections, track_indices, threshold, metric_type, unmatched_detections_in=None):
        if unmatched_detections_in is None:
            detection_indices = list(range(len(detections)))
        else:
            detection_indices = unmatched_detections_in

        if not detection_indices or not track_indices:
            return [], track_indices, detection_indices

        if metric_type == "iou":
            cost_matrix = iou_cost(self.tracks, detections, track_indices, detection_indices)
        else: # cosine
            features = np.array([detections[i].feature for i in detection_indices])
            targets = np.array([self.tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches, unmatched_tracks, unmatched_detections = [], [], []
        
        matched_track_indices = set()
        matched_det_indices = set()
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < threshold:
                track_idx = track_indices[r]
                det_idx = detection_indices[c]
                matches.append((track_idx, det_idx))
                matched_track_indices.add(track_idx)
                matched_det_indices.add(det_idx)
        
        unmatched_tracks = [idx for idx in track_indices if idx not in matched_track_indices]
        unmatched_detections = [idx for idx in detection_indices if idx not in matched_det_indices]
        
        return matches, unmatched_tracks, unmatched_detections

class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.extractor = Extractor(model_path, use_cuda=use_cuda)
        metric = NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
        self.width = self.height = 0

    def update(self, bbox_xywh, confidences, classes, ori_img):
        self.height, self.width = ori_img.shape[:2]
        
        # generate detections
        features = self.extractor(self._get_crops(bbox_xywh, ori_img))
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences) if conf > self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            box = track.to_tlwh()
            x1, y1, w, h = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)
            
            track_id = track.track_id
            
            # Find the original class_id for this track
            # This is a simple but effective way: find the detection that overlaps most with the track
            best_iou = 0
            cls_id = -1
            for i in range(len(bbox_tlwh)):
                # Convert original detection to tlbr
                det_box = bbox_tlwh[i]
                iou_val = iou(box, np.array([det_box]))[0]
                if iou_val > best_iou:
                    best_iou = iou_val
                    cls_id = classes[i]

            outputs.append([x1, y1, x2, y2, track_id, cls_id])
        if len(outputs) > 0:
            return np.array(outputs)
        return np.array([])
    
    def _xywh_to_tlwh(self, bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone().numpy()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _get_crops(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x, y, w, h = box
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.width, x2)
            y2 = min(self.height, y2)
            if x1 >= x2 or y1 >= y2:
                continue
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        return im_crops