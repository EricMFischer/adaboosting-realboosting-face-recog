from abc import ABC, abstractmethod
import numpy as np
from joblib import Parallel, delayed

class Weak_Classifier(ABC):
    #initialize a harr filter with the positive and negative rects
    #rects are in the form of [x1, y1, x2, y2] 0-index
    def __init__(self, id, plus_rects, minus_rects, num_bins):
        self.id = id
        self.plus_rects = plus_rects
        self.minus_rects = minus_rects
        self.num_bins = num_bins
        self.activations = None
        self.sorted_activations = None
        self.sorted_act_indices = None

    #take in one integrated image and return the value after applying the image
    #integrated_image is a 2D np array
    #return value is the number BEFORE polarity is applied
    def apply_filter2image(self, integrated_image):
        pos = 0
        for rect in self.plus_rects:
            rect = [int(n) for n in rect]
            pos += integrated_image[rect[3], rect[2]]\
                 + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1])\
                 - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]])\
                 - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
        neg = 0
        for rect in self.minus_rects:
            rect = [int(n) for n in rect]
            neg += integrated_image[rect[3], rect[2]]\
                 + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1])\
                 - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]])\
                 - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
        return pos - neg

    #take in a list of integrated images and calculate values for each image
    #integrated images are passed in as a 3-D np-array
    #calculate activations for all images BEFORE polarity is applied
    #only need to be called once
    def apply_filter(self, integrated_images):
        values = []
        for idx in range(integrated_images.shape[0]):
            values.append(self.apply_filter2image(integrated_images[idx, ...]))
        if (self.id + 1) % 100 == 0:
            print('Weak Classifier No. %d has finished applying' % (self.id + 1))
        return values

    @abstractmethod
    def calc_error(self, weights, labels):
        pass

    @abstractmethod
    def predict_image(self, integrated_image):
        pass

class Ada_Weak_Classifier(Weak_Classifier):
    def __init__(self, id, plus_rects, minus_rects, num_bins):
        super().__init__(id, plus_rects, minus_rects, num_bins)
        self.polarity = None
        self.threshold = None

    # computes error of applying a weak classifier to dataset given current weights
    # can return error and potentially other identifiers of weak classifier
    def calc_error(self, weights, labels):
        feats = np.subtract(self.activations, self.threshold) * self.polarity
        error = self.calc_feat_error(feats, weights, labels)
        return [self.id, error]

    # features, weights, labels: (37194,)
    def calc_feat_error(self, features, weights, labels):
        misclassified_idxs = np.where(np.not_equal(np.sign(features), labels))[0]
        return np.sum(np.array(weights)[misclassified_idxs])

    def calc_alpha(self, error):
        return 1 / 2 * np.log((1 - error) / error)

    def get_thresholds(self):
        min_act = np.min(self.activations)
        max_act = np.max(self.activations)
        return np.linspace(min_act, max_act, self.num_bins)

    def update_threshold(self, weights, labels, init=False):
        # ths = [.00015, .00025, .0005, .001, .002, .003, .0035, .004, .0045, .005] ''
        ths = self.get_thresholds()
        feats = [np.subtract(self.activations, th) for th in ths] # (50, 37194)

        th_errors = [self.calc_feat_error(feat, weights, labels) for feat in feats]
        adj_errors = [error if error < 0.5 else 1 - error for error in th_errors]

        min_error_i = adj_errors.index(min(adj_errors))
        self.threshold = ths[min_error_i]
        if init:
            self.polarity = -1 if th_errors[min_error_i] > 0.5 else 1

    '''
    # iterate through activations in sorted order, and find e = min(BG + (AFS-FS), FS + (ABG-BG))
    # reason for 2 values: choosing first term as threshold makes it such that values above that
    # response level are considered positive (BG is false positives, AFS-FS is false negatives),
    # and with second term values below are considered positive (ABG-BG is false positives, FS
    # is false negatives)
    # AFS: sum of weights of face samples
    # ABG: sum of weights of all non-face samples
    # FS: sum of positive weights of face samples so far
    # BG: sum of negative weights of non-face samples so far
    # find minimum value of e and use the feature value of corresponding sample as threshold
    def find_threshold(self, weights, labels):
        AFS = np.sum(weights[:11838]) # sum of weights of all face samples
        ABG = np.sum(weights[11838:]) # sum of weights of all non-face samples

        FS, BG = 0, 0
        error = 1000000
        feat_val_to_use_as_threshold = None
        for i, act in enumerate(self.sorted_activations):
            orig_act_i = self.sorted_act_indices[i]

            e = min(BG + (AFS-FS), FS + (ABG-BG))
            if e < error:
                feat_val_to_use_as_threshold = self.activations[orig_act_i]

            act_weight = weights[orig_act_i]
            act_label = labels[orig_act_i]
            # add to FS if ex is face sample, or BG if non-face sample
            if act_label is 1:
                FS += act_weight
            else:
                BG += act_weight

        return feat_val_to_use_as_threshold

    def update_threshold_viola_jones(self, weights, labels, init=False):
        # np.argsort: if an array were sorted, gives you ordered indices of original array
        # ex: for x = [3, 1, 2], np.argsort(x) = [1, 2, 0]
        if init:
            self.sorted_act_indices = np.argsort(self.activations)
            self.sorted_activations = np.array(self.activations)[self.sorted_act_indices]

        self.threshold = self.find_threshold(weights, labels)
        feat = np.subtract(self.activations, self.threshold)

        th_error = self.calc_feat_error(feat, weights, labels)
        self.polarity = -1 if th_error > 0.5 else 1
    '''

    def predict_image(self, integrated_image):
        value = self.apply_filter2image(integrated_image)
        return self.polarity * np.sign(value - self.threshold)

# each wc has all data samples as scalars spread out across its bins
class Real_Weak_Classifier(Weak_Classifier):
    def __init__(self, id, plus_rects, minus_rects, num_bins):
        super().__init__(id, plus_rects, minus_rects, num_bins)
        self.thresholds = None # upper edges of bins
        # bin_pqs: 2 arrays of ps and qs (qty = num_bins), changes when weights are updated
        self.bin_pqs = None
        # array of 37914 indicating bin index of scalar activations in order for each example x_i
        self.train_assignment = []

    def init_bins(self, weights, labels):
        min_act, max_act = min(self.activations), max(self.activations)
        bin_width = (max_act - min_act) / self.num_bins
        # creates array with 25 numbers representing upper thresholds
        self.thresholds = np.linspace(min_act + bin_width, max_act, num=self.num_bins)
        for activation in self.activations:
            bin_idx = np.sum(self.thresholds < activation)
            self.train_assignment.append(bin_idx)
        self.set_bin_pqs(weights, labels)

    def set_bin_pqs(self, weights, labels):
        p = np.zeros(self.num_bins) # sum of weights of positive samples at each bin b
        q = np.zeros(self.num_bins) # sum of weights of negative samples at each bin b
        # enumerate thru train_assignment size 37914 indicating bin index of each example x_i
        for i, bin_idx in enumerate(self.train_assignment):
            if labels[i] is 1:
                p[bin_idx] += weights[i]
            else:
                q[bin_idx] += weights[i]
        self.bin_pqs = [p, q]

    # If you reuse Adaboost weak classifiers, don't have to calc errors, only update weights
    def calc_error(self, weights, labels, t):
        return None

    def predict_image(self, integrated_image):
        value = self.apply_filter2image(integrated_image)
        bin_idx = np.sum(self.thresholds < value)
        p = self.bin_pqs[0, bin_idx]
        p = 0.0001 if p is 0 else p
        q = self.bin_pqs[1, bin_idx]
        q = 0.0001 if q is 0 else q
        return 0.5 * np.log(p / q)

def main():
    plus_rects = [(1, 2, 3, 4)]
    minus_rects = [(4, 5, 6, 7)]
    num_bins = 50
    ada_hf = Ada_Weak_Classifier(plus_rects, minus_rects, num_bins)
    real_hf = Real_Weak_Classifier(plus_rects, minus_rects, num_bins)

if __name__ == '__main__':
    main()
