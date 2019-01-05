import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

import cv2
from weak_classifier import Ada_Weak_Classifier, Real_Weak_Classifier
from im_process import image2patches, nms, normalize
from utils import *

class Boosting_Classifier:
    def __init__(self, haar_filters, data, labels, num_chosen_wc, num_bins, visualizer, num_cores, style):
        self.filters = haar_filters
        self.data = data
        self.labels = labels
        self.num_chosen_wc = num_chosen_wc
        self.num_bins = num_bins
        self.visualizer = visualizer
        self.num_cores = num_cores
        self.style = style
        self.chosen_wcs = []
        if style == 'Ada':
            self.weak_classifiers = [Ada_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
                                     for i, filt in enumerate(self.filters)]
        elif style == 'Real':
            self.weak_classifiers = [Real_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
                                     for i, filt in enumerate(self.filters)]

    def calculate_training_activations(self, save_dir = None, load_dir = None):
        print('Calculating activations for %d weak classifiers, using %d images.' % (len(self.weak_classifiers), self.data.shape[0]))
        if load_dir is not None and os.path.exists(load_dir):
            print('[Finding cached activations in %s]' % load_dir)
            wc_activations = np.load(load_dir) # (10032, 37194)
        else:
            if self.num_cores == 1:
                wc_activations = [wc.apply_filter(self.data) for wc in self.weak_classifiers]
            else:
                wc_activations = Parallel(n_jobs = self.num_cores)(delayed(wc.apply_filter)(self.data) for wc in self.weak_classifiers)
            wc_activations = np.array(wc_activations)
            if save_dir is not None:
                np.save(save_dir, wc_activations)
                print('[Saved calculated activations to %s]' % save_dir)
        for wc in self.weak_classifiers:
            wc.activations = wc_activations[wc.id, :]
        return wc_activations

    def train(self, t = 0, wts = None):
        # potential conditions to halt adaboost algorithm:
        # training error of strong classifier H(x) is below a threshold or almost 0
        # all remaining weak classifiers have error close to 0.5 and are redundant
        wts = self.init_wts() if wts is None else wts
        checkpoints = [0, 10, 50, 100, 125, 150, 160, 175, 200]
        while t <= 201: # self.num_chosen_wc
            # each step t, compute weighted error for each wc
            wc_ids_errors = self.calc_wc_errors(wts, t) if self.style is 'Ada' else None
            if t in checkpoints:
                self.cache_sc(t)
            self.cache_data(wc_ids_errors, wts, t)

            # choose wc with min weighted error
            alpha_and_wc = self.get_best_wc(wc_ids_errors, t)
            print('got best wc for realboosting: ', alpha_and_wc)
            self.chosen_wcs.append(alpha_and_wc)

            # after choosing wc, update weights of data points
            # weights of incorrectly classified data increase and correctly classified
            # decrease, so incorrectly classified receive more "attention” in next run
            wts = self.update_weights(wts, alpha_and_wc)
            if self.style is 'Ada':
                self.update_wc_thresholds(wts)
            else:
                self.update_bin_pqs(wts)
            t += 1

    def cache_data(self, wc_ids_errors, weights, t):
        prefix = 'real_' if self.style is not 'Ada' else 'ada_'

        sc_scores = self.get_ada_or_real_sc_score()
        save_data(sc_scores, './{}sc_scores/sc_scores_step_{}.pkl'.format(prefix, t))

        tr_acc = np.mean(np.sign(sc_scores) == self.labels)
        save_data(tr_acc, './{}sc_accuracy/sc_accuracy_step_{}.pkl'.format(prefix, t))
        print('Start of step %d SC training accuracy is: ' % t, tr_acc)

        if self.style is 'Ada':
            save_data(self.chosen_wcs, './{}chosen_wcs/chosen_wcs_step_{}.pkl'.format(prefix, t))
            print('[Saved chosen weak classifiers at start of step %d]' % t)

            save_data(wc_ids_errors, './{}wcs/wc_ids_errors_step_{}.pkl'.format(prefix, t))
            print('[Saved weak classifier ids and errors at start of step %d]' % t)
            print('Mean of weak classifier errors: ', np.mean(wc_ids_errors[:, 1]))

            save_data(weights, './{}weights/weights_step_{}.pkl'.format(prefix, t))
            print('[Saved data weights at start of step %d]' % t)

    def get_ada_or_real_sc_score(self):
        if self.style is 'Ada':
            return [self.sc_function(self.data[i, ...]) for i in range(self.data.shape[0])]
        else:
            return [self.real_sc_func(self.data[i, ...]) for i in range(self.data.shape[0])]

    def cache_sc(self, t):
        prefix = 'real_' if self.style is not 'Ada' else 'ada_'
        save_data(self, './{}sc/sc_step_{}.pkl'.format(prefix, t))
        print('[Saved strong classifier at start of step %d]' % t)

    def init_wts(self):
        num_data_pts = len(self.data)
        init_wt = 1 / num_data_pts
        return np.array([init_wt] * num_data_pts).T

    def init_ada_wc_props(self):
        for wc in self.weak_classifiers:
            wc.update_threshold(self.init_wts(), self.labels, True)

    def init_real_wc_props(self):
        for wc in self.weak_classifiers:
            wc.init_bins(self.init_wts(), self.labels)

    def calc_wc_errors(self, weights, t):
        ids_errors = Parallel(n_jobs = self.num_cores)(delayed(wc.calc_error)(weights, self.labels) for wc in self.weak_classifiers)
        return np.array(ids_errors)

    # returns the alpha and wc with the min weighted error. as data samples change
    # their weights over time, the histograms and threshold theta will change.
    def get_best_wc(self, wc_ids_errors, t):
        if self.style is not 'Ada':
            print('getting best wc for realboosting')
            return get_data('./chosen_wcs/chosen_wcs_step_100.pkl')[t]
        min_wc_error = np.min(wc_ids_errors[:, 1])
        error_i = list(wc_ids_errors[:, 1]).index(min_wc_error)
        wc_id = wc_ids_errors[:, 0][error_i]
        for wc in self.weak_classifiers:
            if wc.id == wc_id:
                print('Adding wc id: ', wc.id)
                return [wc.calc_alpha(min_wc_error), wc]

    # In training, you don't need to call predict_image directly. As activations
    # don't change in the training process they are the only thing you need.
    # In other words, when you need weak classifier predictions for the training
    # images, get them from the activation member variable.
    def update_weights(self, weights, alpha_and_wc): # returns vector (37194,)
        if self.style is 'Ada':
            wc_preds = self.calc_ada_wc_predictions(alpha_and_wc[1])
            exp_term = np.exp(-1 * np.multiply(self.labels, wc_preds) * alpha_and_wc[0])
        else:
            wc_preds = self.calc_real_wc_predictions(alpha_and_wc[1], weights)
            exp_term = np.exp(-1 * np.multiply(self.labels, wc_preds))

        new_wts = weights * exp_term
        Z_norm = np.sum(new_wts)
        new_wts_norm = np.divide(new_wts, Z_norm)
        print('alpha of chosen wc: ', alpha_and_wc[0])
        return new_wts_norm

    # discrete function h_t(x_i)
    def calc_ada_wc_predictions(self, wc):
        return wc.polarity * np.sign(np.subtract(wc.activations, wc.threshold))

        # continuous function h_t,b(x_i)
    def calc_real_wc_predictions(self, ada_wc, weights):
        # find real wc from loaded ada wc id
        for real_wc in self.weak_classifiers:
            if real_wc.id == ada_wc.id:
                wc = real_wc
                break
        wc_preds = [self.get_real_prediction(wc, wc_act) for wc_act in wc.activations]
        return np.array(wc_preds)

    def get_real_prediction(self, wc, wc_act):
        bin_idx = np.sum(wc.thresholds < wc_act)
        p = wc.bin_pqs[0, bin_idx]
        p = 0.0001 if p is 0 else p
        q = wc.bin_pqs[1, bin_idx]
        q = 0.0001 if q is 0 else q
        return 0.5 * np.log(p / q)

    def update_wc_thresholds(self, weights):
        # after updating weights, update threshold values choosing
        # one with min training error given current data weights
        for wc in self.weak_classifiers:
            wc.update_threshold(weights, self.labels)

    def update_bin_pqs(self, weights):
        # update p_t and q_t values at each step after updating weights
        for wc in self.weak_classifiers:
            wc.set_bin_pqs(weights, self.labels)

    def sc_function(self, image):
        chosen_wcs = get_data('./chosen_wcs/chosen_wcs_step_160.pkl') # temporary for hard mining
        array = [alpha * wc.predict_image(image) for alpha, wc in chosen_wcs] # self.chosen_wcs
        return np.sum([np.array(array)])

    def real_sc_func(self, image):
        array = [wc.predict_image(image) for alpha, wc in self.chosen_wcs]
        return np.sum([np.array(array)])

    def load_trained_wcs(self, save_dir):
        self.chosen_wcs = get_data(save_dir)

    def perform_face_detection(self):
        orig_img = cv2.imread('./Testing_Images/Face_1.jpg', cv2.IMREAD_GRAYSCALE) # CHANGE FILE
        result_img = self.face_detection(orig_img)
        cv2.imwrite('Result_Img_Face_1_with_negatives.png', result_img) # CHANGE FILE

    # applies strong classifier to 16x16-pixel image patches to see if face in patch
    # can compress images to smaller resolution than 1280 X 960
    def face_detection(self, img, scale_step = 20):
        train_preds = []
        for i in range(self.data.shape[0]):
            train_preds.append(self.sc_function(self.data[i, ...]))
        tr_acc = np.mean(np.sign(train_preds) == self.labels)
        print('Check on training accuracy is: ', tr_acc)

        pp = 'pos_preds_xyxy_face_1_with_negatives.pkl' # CHANGE FILE
        if os.path.exists(pp):
            pos_preds_xyxy = get_data(pp)
        else:
            scales = 1 / np.linspace(1, 8, scale_step)
            patches, patches_xyxy = image2patches(scales, img)
            print('Face Detection in Progress..., total %d patches' % patches.shape[0])

            preds = [self.sc_function(patch) for patch in tqdm(patches)]
            pos_preds = np.array(preds) > 0
            print('positive detections mean and sum: ', np.mean(pos_preds), np.sum(pos_preds))

            pos_preds_xyxy = np.array([patches_xyxy[i] + [score] for i, score in enumerate(preds) if score > 0])
            save_data(pos_preds_xyxy, 'pos_preds_xyxy_face_1_with_negatives.pkl') # CHANGE FILE
        if pos_preds_xyxy.shape[0] == 0:
            return

        print('num positive detections before nms: ', pos_preds_xyxy.shape[0])
        pos_preds_xyxy = nms(pos_preds_xyxy, 0.01)
        print('num positive detections after nms: ', pos_preds_xyxy.shape[0])

        for i in range(pos_preds_xyxy.shape[0]):
            pred = pos_preds_xyxy[i, :]
            xy1 = (int(pred[0]), int(pred[1]))
            xy2 = (int(pred[2]), int(pred[3]))
            cv2.rectangle(img, xy1, xy2, (0, 255, 0), 2)

        return img

    # Perform hard negatives mining. You are given background images without
    # faces. Run your strong classifier on these images. Any “faces” detected by your
    # classifier are called “hard negatives”. Add them to the negative population in
    # the training set and re-train your model.
    def get_hard_negative_patches(self, img, scale_step = 10):
        scales = 1 / np.linspace(1, 8, scale_step)
        patches, patches_xyxy = image2patches(scales, img)
        print('Get Hard Negative in Progress..., total %d patches' % patches.shape[0])

        preds = np.array([self.sc_function(patch) for patch in tqdm(patches)])
        wrong_patches = patches[np.where(preds > 0), ...]

        # apply nms
        # hard_negatives_xyxy = np.array([patches_xyxy[i] + [score] for i, score in enumerate(preds) if score > 0])
        # print('num hard negative detections before nms: ', hard_negatives_xyxy.shape[0])
        # hard_negatives_xyxy = nms(hard_negatives_xyxy, 0.01)
        # print('num hard negative detections after nms: ', hard_negatives_xyxy.shape[0])
        # wrong_patches = np.delete(hard_negatives_xyxy, 4, 1)

        return wrong_patches

    def get_negative_patches(self):
        images = []
        for i in range(1, 4):
            images.append(cv2.imread('./Testing_Images/Non_Face_%d.jpg' % i, cv2.IMREAD_GRAYSCALE))
        # ii = integrate_images(normalize(images))
        return self.get_hard_negative_patches(images[0]) # CHANGE

    def get_top_1000_wc_accuracies(self): # CHANGE FILES
        errors_0_steps = sorted(get_data('./ada_wcs/wc_ids_errors_step_0.pkl')[:, 1])[:1000]
        errors_10_steps = sorted(get_data('./ada_wcs/wc_ids_errors_step_10.pkl')[:, 1])[:1000]
        errors_50_steps = sorted(get_data('./ada_wcs/wc_ids_errors_step_50.pkl')[:, 1])[:1000]
        errors_100_steps = sorted(get_data('./ada_wcs/wc_ids_errors_step_100.pkl')[:, 1])[:1000]
        return {
            0: np.subtract(1, errors_0_steps),
            10: np.subtract(1, errors_10_steps),
            50: np.subtract(1, errors_50_steps),
            100: np.subtract(1, errors_100_steps)
        }

    def get_sc_scores(self):
        return {
            10: get_data('./ada_sc_scores/sc_scores_step_10.pkl'), # length 37194, CHANGE FILES
            50: get_data('./ada_sc_scores/sc_scores_step_50.pkl'),
            100: get_data('./ada_sc_scores/sc_scores_step_100.pkl')
        }

    def get_real_sc_scores(self):
        return {
            10: get_data('./sc_scores/sc_scores_step_10.pkl'), # change
            50: get_data('./sc_scores/sc_scores_step_50.pkl'),
            100: get_data('./sc_scores/sc_scores_step_10.pkl')
        }

    def get_sc_accuracies(self):
        accuracies = []
        for i in range(0, self.num_chosen_wc + 1):
            accuracies.append(get_data('./sc_accuracy/sc_accuracy_step_%d.pkl' % i))
        return accuracies

    def visualize(self):
        if self.style is not 'Ada':
            self.visualizer.labels = self.labels
            self.visualizer.strong_classifier_scores = self.get_real_sc_scores()
            self.visualizer.draw_histograms()
            self.visualizer.draw_rocs()
        else:
            # note: source of data is self.chosen_wcs except for weak_classifier_accuracies
            self.visualizer.labels = self.labels
            self.visualizer.weak_classifier_accuracies = self.get_top_1000_wc_accuracies()
            self.visualizer.strong_classifier_scores = self.get_sc_scores()
            self.visualizer.draw_haar_filters(self.chosen_wcs)
            self.visualizer.draw_sc_accuracies(self.get_sc_accuracies())
            self.visualizer.draw_wc_accuracies()
            self.visualizer.draw_histograms()
            self.visualizer.draw_rocs()
