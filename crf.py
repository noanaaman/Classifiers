import numpy as np
from random import shuffle, seed

class CRF(object):

    def __init__(self, label_codebook, feature_codebook):
        self.label_codebook = label_codebook
        self.feature_codebook = feature_codebook
        num_labels = len(self.label_codebook)
        num_features = len(self.feature_codebook)
        self.feature_parameters = np.zeros((num_labels, num_features))
        self.transition_parameters = np.zeros((num_labels, num_labels))
        
        self.index2Label = {self.label_codebook[label]:label for label in self.label_codebook}

    def train(self, training_set, dev_set):
        """Training function

        Feel free to adjust the hyperparameters (learning rate and batch sizes)
        """
        self.train_sgd(training_set, dev_set, 0.001, 10)

    def train_sgd(self, training_set, dev_set, learning_rate, batch_size):
        """Minibatch SGF for training linear chain CRF

        This should work. But you can also implement early stopping here
        i.e. if the accuracy does not grow for a while, stop.
        """
        num_labels = len(self.label_codebook)
        num_features = len(self.feature_codebook)

        num_batches = len(training_set) / batch_size
        total_expected_feature_count = np.zeros((num_labels, num_features))
        total_expected_transition_count = np.zeros((num_labels, num_labels))

        best_so_far_features = np.zeros((num_labels, num_features))
        best_so_far_transitions = np.zeros((num_labels, num_labels))
        
        accuracy = sequence_accuracy(self, dev_set)
        print(accuracy)
        converged = False
        runs_since_update = 0

        while not converged:
            shuffle(training_set)
            updated = False
            for j in range(num_batches):
                batch = training_set[j*batch_size:(j+1)*batch_size]
                total_expected_feature_count.fill(0)
                total_expected_transition_count.fill(0)
                total_observed_feature_count, total_observed_transition_count = self.compute_observed_count(batch)
                
                for sequence in batch:
                    transition_matrices = self.compute_transition_matrices(sequence)
                    alpha_matrix = self.forward(sequence, transition_matrices)
                    beta_matrix = self.backward(sequence, transition_matrices)
                    expected_feature_count, expected_transition_count = \
                            self.compute_expected_feature_count(sequence, alpha_matrix, beta_matrix, transition_matrices)
                    total_expected_feature_count += expected_feature_count
                    total_expected_transition_count += expected_transition_count

                feature_gradient = (total_observed_feature_count - total_expected_feature_count) / len(batch)
                transition_gradient = (total_observed_transition_count - total_expected_transition_count) / len(batch)

                self.feature_parameters += learning_rate * feature_gradient
                self.transition_parameters += learning_rate * transition_gradient

                new_accuracy = sequence_accuracy(self, dev_set)

                if new_accuracy > accuracy:
                    print(new_accuracy)
                    best_so_far_features = self.feature_parameters.copy()
                    best_so_far_transitions = self.transition_parameters.copy()
                    accuracy = new_accuracy
                    updated = True

            print('finished epoch')
            if updated:
                runs_since_update = 0
            else:
                runs_since_update += 1

            if (runs_since_update > 3):
                converged = True

        
        self.feature_parameters = best_so_far_features.copy()
        self.transition_parameters = best_so_far_transitions.copy()
        

    def compute_transition_matrices(self, sequence):
        """Compute transition matrices (denoted as M on the slides)

        Compute transition matrix M for all time steps.

        We add one extra dummy transition matrix at time 0. 
        This one dummy transition matrix should not be used ever, but it 
        is there to make the index consistent with the slides

        The matrix for the first time step does not use transition features
        and should be a diagonal matrix.

        Returns :
            a list of transition matrices
        """
        transition_matrices = []
        num_labels = len(self.label_codebook)
        transition_matrix = np.zeros((num_labels, num_labels))
        transition_matrices.append(transition_matrix)
        for t in range(len(sequence)):
            transition_matrix = np.zeros((num_labels, num_labels))
            # M_0,1 is diagonal and only uses the parameters of the feature functions
            if t==0:
                for i in range(num_labels):
                    lambdas = sum([self.feature_parameters[i][feature] for feature in sequence[t].feature_vector])
                    transition_matrix[i][i] = np.exp(lambdas)
            # for other transitions we need to compute exp(lambda(y1,y2)+lambda(y2,f1)+...+lambda(y2,FN)) for each feature f in the data at t
            else:
                for i in range(num_labels):
                    for j in range(num_labels):
                        lambdas = self.transition_parameters[i][j]
                        lambdas += sum([self.feature_parameters[j][feature] for feature in sequence[t].feature_vector])
                        transition_matrix[i][j] = np.exp(lambdas)
            transition_matrices.append(transition_matrix)
        return transition_matrices

    def forward(self, sequence, transition_matrices):
        """Compute alpha matrix in the forward algorithm
        """
        num_labels = len(self.label_codebook)
        alpha_matrix = np.zeros((len(sequence) + 1,num_labels))
        for t in range(len(sequence) + 1):
            if t==0:
                alpha_matrix[t] = 1
            else:
                alpha_matrix[t] = alpha_matrix[t-1].dot(transition_matrices[t])
        return np.transpose(alpha_matrix)            

    def backward(self, sequence, transition_matrices):
        """Compute beta matrix in the backward algorithm

        TODO: Implement this function
        """
        num_labels = len(self.label_codebook)
        beta_matrix = np.zeros((len(sequence) + 1,num_labels))
        time = range(len(sequence) + 1)
        time.reverse()
        for t in time:
            if t==len(sequence):
                beta_matrix[t] = 1
            else:
                beta_matrix[t] = transition_matrices[t+1].dot(beta_matrix[t+1])
        return np.transpose(beta_matrix)

    def decode(self, sequence):
        """Find the best label sequence from the feature sequence

        TODO: Implement this function

        Returns :
            a list of label indices (the same length as the sequence)
        """
        backtrace_pointers = self.viterbi(sequence)

        best_sequence = []

        t = len(sequence)-1
        current = int(backtrace_pointers[len(self.label_codebook)][len(sequence)-1])
        
        #go backwards and get the pervious state
        while len(best_sequence)<len(sequence):
            best_sequence.append(current)
            current = int(backtrace_pointers[current][t])
            t -= 1

        #reverse the order and replace the index with the label
        best_sequence = [best_sequence[i] for i in xrange(len(best_sequence)-1,-1,-1)]
        
        return best_sequence


    def viterbi(self,sequence):
        
        trellis = np.zeros((len(self.label_codebook)+1,len(sequence)))
        backtrace_pointers = np.zeros((len(self.label_codebook)+1,len(sequence)))

        # initialization step:
        for l in range(len(self.label_codebook)):
            lambdas = sum([self.feature_parameters[l][feature] for feature in sequence[0].feature_vector])
            trellis[l][0] = np.exp(lambdas)

        # recursion step
        for t in range(1,len(sequence)):
            for l in range(len(self.label_codebook)):
                lambdas = sum([self.feature_parameters[l][feature] for feature in sequence[t].feature_vector])
                max_arg = max([(prev_state,trellis[prev_state][t - 1] * np.exp(self.transition_parameters[prev_state][l]+lambdas)) for prev_state in range(len(self.label_codebook))],key=lambda x: x[1])
                trellis[l][t] = max_arg[1]
                backtrace_pointers[l][t] = max_arg[0]

        # termination step
        term_max_arg = max([(last_state,trellis[last_state][len(sequence)-1]) for last_state in range(len(self.label_codebook))], key=lambda x: x[1])
        backtrace_pointers[len(self.label_codebook)][len(sequence)-1] = term_max_arg[0]

        return backtrace_pointers

    def compute_observed_count(self, sequences):
        """Compute observed counts of features from the minibatch

        Returns :
            A tuple of
                a matrix of feature counts 
                a matrix of transition-based feature counts
        """
        num_labels = len(self.label_codebook)
        num_features = len(self.feature_codebook)

        feature_count = np.zeros((num_labels, num_features))
        transition_count = np.zeros((num_labels, num_labels))
        for sequence in sequences:
            for t in range(len(sequence)):
                if t > 0:
                    transition_count[sequence[t-1].label_index, sequence[t].label_index] += 1
                feature_count[sequence[t].label_index, sequence[t].feature_vector] += 1
        return feature_count, transition_count

    def compute_expected_feature_count(self, sequence, 
            alpha_matrix, beta_matrix, transition_matrices):
        """Compute expected counts of features from the sequence

        Returns :
            A tuple of
                a matrix of feature counts 
                a matrix of transition-based feature counts
        """
        num_labels = len(self.label_codebook)
        num_features = len(self.feature_codebook)

        feature_count = np.zeros((num_labels, num_features))
        sequence_length = len(sequence)
        Z = np.sum(alpha_matrix[:,-1])

        #gamma = alpha_matrix * beta_matrix / Z 
        gamma = np.exp(np.log(alpha_matrix) + np.log(beta_matrix) - np.log(Z))
        for t in range(sequence_length):
            for j in range(num_labels):
                feature_count[j, sequence[t].feature_vector] += gamma[j, t]

        transition_count = np.zeros((num_labels, num_labels))
        for t in range(sequence_length - 1):
            transition_count += (transition_matrices[t] * np.outer(alpha_matrix[:, t], beta_matrix[:,t+1])) / Z
        return feature_count, transition_count

def sequence_accuracy(sequence_tagger, test_set):
    correct = 0.0
    total = 0.0
    for sequence in test_set:
        decoded = sequence_tagger.decode(sequence)
        assert(len(decoded) == len(sequence))
        total += len(decoded)
        for i, instance in enumerate(sequence):
            if instance.label_index == decoded[i]:
                correct += 1
    return correct / total


