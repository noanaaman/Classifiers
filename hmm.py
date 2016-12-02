"""Hidden Markov Model sequence tagger

"""

import numpy, nltk
from _collections import defaultdict

from classifier import Classifier

class HMM(Classifier):
    def get_model(self): return None
    def set_model(self, model): pass
    model = property(get_model, set_model)

    def __init__(self):
        self.transition_count_table = defaultdict(nltk.FreqDist)
        self.feature_count_table = defaultdict(nltk.FreqDist)
        self.all_observations = set()
        self.all_states = set()
        self.transition_matrix = {}
        self.emission_matrix = {}

    def _collect_counts(self, instance_list):
        for instance in instance_list:
            previous = 'START'
            features = instance.features()
            #attach the tag for each observation in the sequence
            data = [(features[i],instance.label[i]) for i in range(0,len(features))]
            for observation,tag in data:
                self.all_observations.add(observation)
                self.all_states.add(tag)
                self.feature_count_table[tag][observation] += 1
                self.transition_count_table[previous][tag] += 1
                previous = tag

            self.transition_count_table[previous]['END'] += 1

        #Laplace smoothing
        for state in self.all_states:
            for observation in self.all_observations:
                self.feature_count_table[state][observation] += 1

        for state in self.all_states:
            for next_state in self.all_states:
                self.transition_count_table[state][next_state] += 1

        #Add start and end states
        for state in self.all_states:
            self.transition_count_table['START'][state] += 1
            self.transition_count_table[state]['END'] += 1

    def train(self, instance_list):
        print('collecting counts')
        self._collect_counts(instance_list)
        self.label2index = {}
        i = 0

        for state in self.all_states:
            self.label2index[state] = i
            i += 1

        self.index2label = {self.label2index[state]:state for state in self.label2index}

        #turn the counts into probabilities 
        for state in self.transition_count_table:
            total = self.transition_count_table[state].N()
            self.transition_matrix[state] = {nextState : self.transition_count_table[state][nextState]/float(total) for nextState in self.transition_count_table[state].keys()}

        for state in self.all_states:
            total = self.feature_count_table[state].N()
            self.emission_matrix[state] = {observation : self.feature_count_table[state][observation]/float(total) for observation in self.all_observations}
            self.emission_matrix[state]['UNK'] = 1/float(total) #add emission for unseen observations

        print("got probabilities")


    def get_emission(self,state,observation):
        """gets the emission for known observations, or the emission computed after smoothing for unseen ones"""
        if observation in self.emission_matrix[state]:
            return self.emission_matrix[state][observation]

        return self.emission_matrix[state]['UNK']

    def classify(self, instance):
        """Gets the backtrace pointers and creates the sequence of tags in order"""
        features = instance.features()
        data = [(features[i],instance.label[i]) for i in range(0,len(features))]
        backtrace_pointers = self.dynamic_programming_on_trellis(instance, False)[1]
        best_sequence = []

        i = len(data)-1
        current = int(backtrace_pointers[len(self.all_states)][len(data)-1])

        #go backwards and get the pervious state
        while len(best_sequence)<len(data):
            best_sequence.append(current)
            current = int(backtrace_pointers[current][i])
            i -= 1

        #reverse the order and replace the index with the label
        best_sequence = [self.index2label[best_sequence[i]] for i in xrange(len(best_sequence)-1,-1,-1)]
        
        return best_sequence

    def compute_observation_loglikelihood(self, instance):
        """Compute and return log P(X|parameters) = loglikelihood of observations"""
        trellis = self.dynamic_programming_on_trellis(instance, True)[0]

        loglikelihood = trellis[len(self.all_states)][len(instance) - 1]
        return loglikelihood

    def dynamic_programming_on_trellis(self, instance, run_forward_alg=True):
        """implemented based on the psuedo code in the slides. Is the forward flag is false, we just use max instead of sum and add the backtrace."""
        features = instance.features()

        #attach the tag for each observation in the sequence
        data = [(features[i],instance.label[i]) for i in range(0,len(features))]
        trellis = numpy.zeros((len(self.all_states)+1,len(data)))
        backtrace_pointers = numpy.zeros((len(self.all_states)+1,len(data)))

        if run_forward_alg:

            # initialization step:
            for state in self.all_states:
                trellis[self.label2index[state]][0] = self.transition_matrix['START'][state] * self.get_emission(state,data[0][0])

            # recursion step
            for i in range(1, len(data)):
                for state in self.all_states:
                    trellis[self.label2index[state]][i] = self.get_emission(state,data[i][0])
                    trellis[self.label2index[state]][i] *= sum([trellis[self.states_mapping[prev_state]][i - 1] * self.transition_matrix[prev_state][state] for prev_state in self.all_states])

            # termination step
            trellis[len(self.all_states)][len(data) - 1] = sum([trellis[self.states_mapping[prev_state]][len(data) - 1] * self.transition_matrix[prev_state][state] for prev_state in self.all_states])

        else:
            # initialization step:
            for state in self.all_states:
                trellis[self.label2index[state]][0] = self.transition_matrix['START'][state] * self.get_emission(state,data[0][0])
                backtrace_pointers[self.label2index[state]][0] = -1.0

            # recursion step
            for i in range(1, len(data)):
                for state in self.all_states:
                    max_arg = max([(self.label2index[prev_state],trellis[self.label2index[prev_state]][i - 1] * self.transition_matrix[prev_state][state]) for prev_state in self.all_states],key=lambda x: x[1])
                    trellis[self.label2index[state]][i] = self.get_emission(state,data[i][0])*max_arg[1]
                    backtrace_pointers[self.label2index[state]][i] = max_arg[0]

            # termination step
            term_max_arg = max([(self.label2index[prev_state], trellis[self.label2index[prev_state]][len(data)-1] * self.transition_matrix[prev_state]['END']) for prev_state in self.all_states], key= lambda x: x[1])
            backtrace_pointers[len(self.all_states)][len(data)-1] = term_max_arg[0]

        return (trellis, backtrace_pointers)

    def train_semisupervised(self, unlabeled_instance_list, labeled_instance_list=None):
        if labeled_instance_list is not None:
            self.train(labeled_instance_list)

        else:
            #TODO: initialize the model randomly
            pass

        while True:
            #E-Step
            self.expected_transition_counts = numpy.zeros((1,1))
            self.expected_feature_counts = numpy.zeros((1,1))
            for instance in unlabeled_instance_list:
                (alpha_table, beta_table) = self._run_forward_backward(instance)
                #TODO: update the expected count tables based on alphas and betas
                #also combine the expected count with the observed counts from the labeled data
                #M-Step
                #TODO: reestimate the parameters
                # if self._has_converged(old_likelihood, likelihood):\
                # 	break


    def _has_converged(self, old_likelihood, likelihood):
            """Determine whether the parameters have converged or not (EXTRA CREDIT)

		Returns True if the parameters have converged.
		"""
            return True

    def _run_forward_backward(self, instance):
        """Forward-backward algorithm for HMM using trellis (EXTRA CREDIT)

		Fill up the alpha and beta trellises (the same notation as
		presented in the lecture and Martin and Jurafsky)
		You can reuse your forward algorithm here

		return a tuple of tables consisting of alpha and beta tables
		"""
        alpha_table = numpy.zeros((1,1))
        beta_table = numpy.zeros((1,1))
        #TODO: implement forward backward algorithm right here

        return (alpha_table, beta_table)
