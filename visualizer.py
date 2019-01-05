import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image

# make sure to include 3 photos in report
class Visualizer:
	def __init__(self, histogram_intervals, top_wc_intervals):
		self.histogram_intervals = histogram_intervals # not being used
		self.top_wc_intervals = top_wc_intervals # not being used
		self.weak_classifier_accuracies = {}
		self.strong_classifier_scores = {}
		self.labels = None

	# plot the histograms of the positive and negative populations over F(x),
	# for steps t = 10, 50, 100, respectively.
	def draw_histograms(self):
		for step_t in self.strong_classifier_scores:
			scores = self.strong_classifier_scores[step_t]
			pos_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == 1]
			neg_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == -1]

			bins = np.linspace(np.min(scores), np.max(scores), 100)

			plt.figure()
			plt.hist(pos_scores, bins, alpha=0.5, label='Faces')
			plt.hist(neg_scores, bins, alpha=0.5, label='Non-Faces')
			plt.legend(loc='upper right')
			plt.title('Using %d Weak Classifiers in RealBoosting' % step_t) # CHANGE FILE
			plt.savefig('histogram_%d_realboosting.png' % step_t)

	# based on histograms, plot their corresponding ROC curves for t = 10, 50, 100
	def draw_rocs(self):
		plt.figure()
		for step_t in self.strong_classifier_scores:
			scores = self.strong_classifier_scores[step_t]
			fpr, tpr, _ = roc_curve(self.labels, scores)
			plt.plot(fpr, tpr, label = 'No. %d Weak Classifiers' % step_t)
		plt.legend(loc = 'lower right')
		plt.title('ROC Curve for RealBoosting') # CHANGE FILE
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.savefig('ROC Curve RealBoosting')

	# plot training error of strong classifier over number of steps t
	def draw_sc_accuracies(self, sc_accuracies):
		plt.figure()
		plt.plot(sc_accuracies)
		plt.ylabel('Accuracy')
		plt.xlabel('t Steps')
		plt.title('Strong Classifier Accuracy Over t Steps')
		plt.legend(loc = 'upper right')
		plt.savefig('Strong Classifier Accuracy')

	# at steps t = 0, 10, 50, 100, plot the curve for training accuracies of
	# top 1,000 weak classifiers among the pool of remaining weak classifiers
	# in decreasing order. compare these four curves.
	def draw_wc_accuracies(self):
		plt.figure()
		for step_t in self.weak_classifier_accuracies:
			accuracies = self.weak_classifier_accuracies[step_t]
			plt.plot(accuracies, label = 'After %d Selections' % step_t)
		plt.ylabel('Accuracy')
		plt.xlabel('Weak Classifiers')
		plt.title('Top 1000 Weak Classifier Accuracies')
		plt.legend(loc = 'upper right')
		plt.savefig('Top 1000 Weak Classifier Accuracies')

	# display the top 20 Haar filters after boosting
	# report corresponding voting weights {αt: t = 1, ..., 20}.
	# NOTE: need to take the polarity of the corresponding classifier into account
	def draw_haar_filters(self, chosen_wcs):
		voting_wts = []
		for i, alpha_and_wc in enumerate(chosen_wcs[:20]):
			voting_wts.append(alpha_and_wc[0])
			img = np.array(Image.open('./newface16/face16_000001.bmp'), dtype=np.uint8)
			plus_rects = alpha_and_wc[1].plus_rects
			minus_rects = alpha_and_wc[1].minus_rects
			if alpha_and_wc[1].polarity is -1:
				plus_rects, minus_rects = minus_rects, plus_rects

			for plus_rect in plus_rects:
				xy1 = (int(plus_rect[0]), int(plus_rect[1]))
				xy2 = (int(plus_rect[2]), int(plus_rect[3]))
				cv2.rectangle(img, xy1, xy2, (255, 255, 255), -1)
			for minus_rect in minus_rects:
				xy1 = (int(minus_rect[0]), int(minus_rect[1]))
				xy2 = (int(minus_rect[2]), int(minus_rect[3]))
				cv2.rectangle(img, xy1, xy2, (0, 0, 0), -1)

			cv2.imwrite('./haar_filters/haar_filter_%d.png' % (i + 1), img)
		print('Alpha Voting Weights: ', voting_wts)
