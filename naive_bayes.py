# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
from nltk import FreqDist, DictionaryProbDist
import numpy
from collections import defaultdict

class NaiveBayes(Classifier):
    u"""A naïve Bayes classifier."""
    def __init__(self, model={}):
        super(NaiveBayes, self).__init__(model)

    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model
    model = property(get_model, set_model)

    def train(self, instances):
        """Remember the labels associated with the features of instances."""    

        label_counts = FreqDist()
        feature_counts = defaultdict(FreqDist)
        all_features = set()

        #collect counts: C(feature,label) and C(label)
        for instance in instances:
            if instance.label != '': #I'm throwing out one blog without a label that is in the corpus for some reason
                label_counts[instance.label] += 1
                features = instance.features()
                for feature in features:
                    all_features.add(feature)
                    feature_counts[instance.label][feature] += 1

        #smoothing, and also making sure that all features are counted for each label 
        for label in feature_counts.keys():
            for feature in all_features:
                feature_counts[label][feature] += 1

        #P(label)
        total = label_counts.N()
        label_probs = {label : float(label_counts[label])/total for label in label_counts}

        #P(feature|label) as a dictionary of dictionaries- C(feature,label)/SUM(C(feature,label) for all the features)
        feature_probs = {}
        for label in feature_counts:
            total = feature_counts[label].N()
            feature_probs[label] = {feature : float(feature_counts[label][feature])/total for feature in feature_counts[label]}

        #set the model
        self.set_model({"label_probs" : label_probs, "feature_probs" : feature_probs, "all_features" : all_features})

        
    def classify(self, instance):
        """Classify an instance using the features seen during training."""
        #I saved the probabilities in the model, but will compute LE with log(P)
        model = self.get_model()

        possible_labels = {}

        #get P(label)*P(feature1|label)*P(feature2|label)*...*P(featureN|label) for each label using sum of log(P)
        for label in model['label_probs']:
            sum_logP = numpy.log2(model['label_probs'][label]) #get P(label)
            for feature in instance.features():
                if feature in model['all_features']: #get P(feature|label) for each feature
                    sum_logP += numpy.log2(model['feature_probs'][label][feature])
            possible_labels[label] = sum_logP

        #return the label with max value
        max_label = None
        max_prob = possible_labels[possible_labels.keys()[0]]
        for label in possible_labels:
            if possible_labels[label] >= max_prob:
                max_prob = possible_labels[label]
                max_label = label

        return max_label
                
        
