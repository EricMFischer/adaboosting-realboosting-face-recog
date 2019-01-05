from boosting_classifier import Boosting_Classifier
from visualizer import Visualizer
from im_process import normalize
from utils import *

def get_data_and_labels(reload = False):
    if not reload:
        data = get_data('data.pkl') # (37194,16,16)
        labels = get_data('labels.pkl') # (37194,)
    else:
        image_w, image_h = 16, 16
        pos_data_dir, neg_data_dir = './newface16', './nonface16'
        data, labels = load_data(pos_data_dir, neg_data_dir, image_w, image_h, False)
        data = integrate_images(normalize(data))
    return data, labels

def add_hard_negatives_to_data(negative_patches):
    for i, patch in enumerate(negative_patches):
        save_data(patch, './nonface16/negative_patch%d.bmp' % i)

def main():
    # flag for debugging
    flag_subset = False
    boosting_type = 'Real' # 'Real' or 'Ada'
    training_epochs = 100 if not flag_subset else 20
    act_cache_dir = 'wc_activations.npy' if not flag_subset else 'wc_activations_subset.npy'
    act_with_negs_cache_dir = 'wc_activations_with_hard_negatives.npy' if not flag_subset else 'wc_activations_with_hard_negatives_subset.npy'

    # data configurations
    image_w = 16
    image_h = 16
    data, labels = get_data_and_labels()

    # number of bins for boosting
    num_bins = 25

    # number of cpus for parallel computing
    num_cores = 8 if not flag_subset else 1 #always use 1 when debugging

    # create Haar filters (10032,2)
    filters = generate_Haar_filters(4, 4, 16, 16, image_w, image_h, flag_subset)

    # create visualizer to draw histograms, roc curves and best weak classifier accuracies
    drawer = Visualizer([10, 20, 50, 100], [1, 10, 20, 50, 100])

    # create boost classifier with a pool of weak classifiers
    boost = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type)

    # calculate filter values for all training images
    boost.calculate_training_activations(act_cache_dir, act_cache_dir)

    # initialize wc threshold and polarity for adaboosting, or
    # wc thresholds, bin_pqs, and train_assignment for realboosting
    # if boosting_type is 'Ada':
    #     boost.init_ada_wc_props()
    # else:
    #     # boost.init_real_wc_props()
    #     boost = get_data('./real_sc/real_sc_initialized.pkl')

    # boost.train()

    # boost.visualize()

    # boost.perform_face_detection()

    negative_patches = boost.get_negative_patches()
    save_data(negative_patches, 'hard_negative_patches_1.pkl')

    # add hard negatives to negative population in training set ./nonface16
    add_hard_negatives_to_data(negative_patches)

    # re-train model with hard negative image patches added to training set
    # data, labels = get_data_and_labels(True)
    # boost = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type)
    # boost.calculate_training_activations(act_with_negs_cache_dir)
    # boost.init_ada_wc_props()
    # boost.train()

if __name__ == '__main__':
    main()
