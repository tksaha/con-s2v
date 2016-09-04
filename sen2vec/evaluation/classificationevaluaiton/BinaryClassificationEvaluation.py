import numpy as np
import math
from sklearn.metrics import roc_curve,auc 
from sklearn.metrics import average_precision_score


class	BinaryClassifierEvaluation:
	def __init__(self,unlabelled_prediction,unlabelled_labels,cutoff_value,labelled_labels):
		self.unlabelled_prediction = unlabelled_prediction
		self.unlabelled_labels = unlabelled_labels
		self.cutoff_value = cutoff_value
		self.labelled_labels = labelled_labels
		self.scores = [float(pred.strip()) for pred in open(self.unlabelled_prediction)]

		self.total_un = len(self.scores)
		self.unlabelled_labels_list = [float(label.strip()) for label in open(unlabelled_labels)]
		self.labelled_labels_list = [float(label.strip()) for label in open(labelled_labels)]
		self.total_lab = len(self.labelled_labels_list)

		self.tp_lab = 0
		self.tn_lab = 0

		for label in self.labelled_labels_list:
			if label > 0.0:
				self.tp_lab = self.tp_lab +1
			else:
				self.tn_lab = self.tn_lab + 1

		self.tp_un = self.tn_un = self.fp_un = self.fn_un = 0.0
		self.index  = 0
		#print "cutoff" + str(cutoff_value); 	
		for prediction in self.scores:
			if float(prediction) > cutoff_value and self.unlabelled_labels_list[self.index] > 0.0:
				self.tp_un = self.tp_un + 1.0 
			elif float(prediction) <= cutoff_value and self.unlabelled_labels_list[self.index] < 0.0:
				self.tn_un = self.tn_un + 1.0
			elif float(prediction) <= cutoff_value and self.unlabelled_labels_list[self.index] > 0.0:
				self.fn_un = self.fn_un + 1.0 
			else:
				self.fp_un = self.fp_un + 1.0 
			self.index = self.index +1


	# return in percentage
	def get_precision(self):	
		prec = 0.0
		try:
			prec = float(self.tp_un)/(self.tp_un+self.fp_un)
		except:
			pass	
		return prec*100.0
	def get_recall(self):
		recall = 0.0;
		try:
			recall = float(self.tp_un)/(self.tp_un+self.fn_un)
		except:
			pass
		return recall*100.0
	def get_f1(self):
		prec = self.get_precision()/100.0 
		recall = self.get_recall()/100.0
		f1 =0.0
		try:
			f1 = (2 * prec * recall) / (prec + recall)
		except:
			pass
		return f1*100
	def get_auc(self):		
		fpr, tpr, thresholds = roc_curve(self.unlabelled_labels_list,self.scores, pos_label=1)
		auc_ = auc(fpr, tpr)
		return auc_*100.0 
	def get_avgprec(self):
		avgprec = average_precision_score(self.unlabelled_labels_list,self.scores)
		return avgprec * 100.0
	def get_wss(self): 
		value = float(self.tn_un + self.fn_un)/self.total_un
		#http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1447545/
		return (value - (1.0- (self.get_recall()/100.0))) * 100.0
	def get_yield(self):
		yield_ = float(self.tp_lab+self.tp_un)/(self.tp_lab + self.tp_un + self.fn_un)
		return yield_*100.0
	def get_burden(self):
		burden_ = float(self.tp_lab+self.tn_lab+self.tp_un+self.fp_un) / (self.total_un + self.total_lab)
		return burden_*100.0
	def get_utility(self): 
		#http://users.cs.fiu.edu/~lzhen001/activities/KDD_USB_key_2010/docs/p173.pdf
		if self.total_un+ self.total_lab > 10000:
			beta_utility = 9 
		else:
			beta_utility = 19 
		utility = ((beta_utility * (self.get_yield()/100.0)) + (1.0 - (self.get_burden()/100.0)))/ (1.0 + beta_utility)
		return utility * 100.0
  	def get_accuracy(self):
	   	return  (self.tp_un+self.tn_un)/float(self.total_un);
	def get_gmean(self):
                recall_p = self.get_recall();
		recall_n = 0.0; 
		try:
			recall_n = float(self.tn_un)/(self.tn_un + self.fp) ;
		except:
			pass;
		return  math.sqrt(recall_p*recall_n); 
	def get_arithmatic_mean_error(self):
	 	false_neg_frac = float(self.fn_un)/(self.tp_un+self.fn_un); 	
		false_pos_frac = float(self.fp_un)/(self.tn_un+self.fp_un); 
		return  (false_neg_frac + false_pos_frac)/2.0; 
	def get_quadratic_mean_error(self):
		false_neg_frac = float(self.fn_un)/(self.tp_un+self.fn_un); 	
		false_pos_frac = float(self.fp_un)/(self.tn_un+self.fp_un); 
		return  math.sqrt((false_neg_frac*false_neg_frac + false_pos_frac*false_pos_frac)/2.0); 
	
		
