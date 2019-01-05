import pickle
import numpy as np

def save_data(data, filename):
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

def get_data(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def main():
    print('pickle test')
    real_sc = get_data('./real_sc/sc_step_0.pkl')
    print('real_sc: ', real_sc.weak_classifiers[0].thresholds)

    # Print chosen weak classifiers' info
    sc = get_data('./sc/sc_step_160.pkl')
    for i, wc in enumerate(sc.chosen_wcs):
        print('step, alpha, id, polarity, th: ', i, wc[0], wc[1].id, wc[1].polarity, wc[1].threshold)

    # Print sc accuracies for all steps
    print('SC accuracies steps 0-160')
    for i in range(0, 161):
        print(i, get_data('./sc_accuracy/sc_accuracy_step_%d.pkl' % i))

    # Print wc ids and errors at different steps
    wc_ids_errors = np.array(get_data('./wcs/wc_ids_errors_step_160.pkl'))
    print('wc_ids_errors at step 160: ', wc_ids_errors)

if __name__ == '__main__':
    main()
