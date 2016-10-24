# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
from nltk import FreqDist
from collections import defaultdict
from random import shuffle, seed
import numpy, scipy
import matplotlib.pyplot as plt

class MaxEnt(Classifier):

    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model

    model = property(get_model, set_model)

    def train(self, instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""
        
            
        self.train_sgd(instances, dev_instances, 0.001, 100)

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Mini-batch Stochastic Gradient 
        """
        print("started training")
        print("batch size: " + str(batch_size))
        print("learning rate: " + str(learning_rate))
        
        for instance in dev_instances:
            instance.features()
        
        label_mapping, params = self.init_params(train_instances)
        print("got matrix size")
        converged = False

        #run GD once to get initial results
        minibatches = [train_instances[i:i + batch_size] for i in xrange(0, len(train_instances), batch_size)]

        for batch in minibatches:
            gradient = self.compute_gradient(batch, params, label_mapping)
            params += gradient * learning_rate


        log_likelihood = self.model_log(dev_instances, params, label_mapping)
        accuracy = self.accuracy(dev_instances, params, label_mapping)

        #print("likelihood: " + str(log_likelihood))
        best_so_far = params
        runs_since_update = 1
        
        while not converged:
            seed(hash("minibatches"))
            shuffle(train_instances)
            minibatches = [train_instances[i:i + batch_size] for i in xrange(0, len(train_instances), batch_size)]
            
            for batch in minibatches:
                gradient = self.compute_gradient(batch, params, label_mapping)
                params += gradient * learning_rate


            #new_log_likelihood = self.model_log(dev_instances, params, label_mapping)
            new_accuracy = self.accuracy(dev_instances, params, label_mapping)
            #print("likelihood: " + str(new_log_likelihood))
            if (new_accuracy > accuracy):
                best_so_far = params
                #log_likelihood = new_log_likelihood
                accuracy = new_accuracy
                runs_since_update = 0
            else:
                runs_since_update += 1

            if (runs_since_update > 5):
                converged = True


        model = {"params": best_so_far, "label_mapping": label_mapping}
        self.set_model(model)
        print("finished training")
            

    def classify(self, instance):
        model = self.get_model()
        label_mapping = model["label_mapping"]
        params = model["params"]
        instance.features()
        features = instance.feature_vector
        probs = self.conditional_probs(label_mapping, params, features)

        max_arg = probs[probs.keys()[0]]
        max_label = ""
        for label in probs:
            if (probs[label] >= max_arg):
                max_arg = probs[label]
                max_label = label
        return max_label

    #used for evaluating on the dev set during training, when the parameters change and there's no saved model
    def eval_classify(self, label_mapping, params, instance):
        features = instance.feature_vector
        probs = self.conditional_probs(label_mapping, params, features)

        max_arg = probs[probs.keys()[0]]
        max_label = ""
        for label in probs:
            if (probs[label] >= max_arg):
                max_arg = probs[label]
                max_label = label
        return max_label

    def accuracy(self, instances, params, label_mapping):
        correct = [self.eval_classify(label_mapping, params, x) == x.label for x in instances]
        return float(sum(correct)) / len(correct)
            

    def conditional_probs(self, label_mapping, params, feature_vector):
        probs = {}
        scores = []
        denominator = 0.0
        #get the posterior for each label
        for label in label_mapping:
            param_sum = 0.0

            #get the lambdas for the features that are in the feature vector
            for feature in feature_vector:
                param_sum += params[feature][label_mapping[label]]
            probs[label] = param_sum
            scores.append(param_sum)
        probs = {label:numpy.exp(probs[label] - scipy.misc.logsumexp(scores)) for label in probs}
        return probs    

    #Used to follow the training process    
    def model_log(self, instances, params, label_mapping):
        likelihood = 0.0
        for instance in instances:
            features = instance.feature_vector
            probs = self.conditional_probs(label_mapping, params, features)
            likelihood += numpy.log2(probs[instance.label])

        return likelihood

    def compute_gradient(self, batch, params, label_mapping):
        observed = self.observed(batch, params, label_mapping)
        expected = self.expected(batch, params, label_mapping)

        return observed - expected

    def observed(self, batch, params, label_mapping):
        observed = numpy.zeros(params.shape)

        #get all the labels and counts of features
        for instance in batch:
            features = instance.feature_vector
            for feature in features:
                observed[feature][label_mapping[instance.label]] += 1

        return observed

    def expected(self, batch, params, label_mapping):
        #each cell represents a (feature,label) pair
        expected = numpy.zeros(params.shape)

        #Go through each document, get its conditional probabilities for each label and add it to the sum for each relevant(feature,label) pair
        for instance in batch:
            features = instance.feature_vector
            probs = self.conditional_probs(label_mapping, params, features)
            for label in label_mapping:
                for feature in features:
                    expected[feature][label_mapping[label]] += probs[label]
        
        return expected

    def init_params(self, train_instances):
        labels = set()

        #get all the labels and extract features
        for instance in train_instances:
            instance.features()
            labels.add(instance.label)

        #map labels to integer IDs
        label_mapping = {}
        label_id = 0
        for label in labels:
            label_mapping[label] = label_id
            label_id += 1

        #the number of features will be the number of rows in the observed counts matrix. The last feature for any type of document is the bias and will give us the total number of features in the model
        total_features = max(train_instances[0].feature_vector) + 1

        #represent the observed counts in a matrix of the right dimentions 
        params = numpy.zeros((total_features, len(label_mapping)))

        print("labels: " + str(len(label_mapping)))
        print("features: " + str(total_features))
        return label_mapping, params

    
