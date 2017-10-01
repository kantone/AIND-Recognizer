import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_bic = float("inf")

        for num_components in range(self.min_n_components, self.max_n_components + 1):
            candidate_model = self.base_model(num_components)

            try:
                logL = candidate_model.score(self.X, self.lengths)
                num_samples, num_features = self.X.shape
                num_params = num_components ** 2 + 2 * num_components * num_features - 1
                bic = -2 * logL + num_params * np.log(num_samples)

                if bic < best_bic:
                    best_model = candidate_model
                    best_bic = bic
            except ValueError:
                if self.verbose:
                    print("failure to calculate BIC with {} components".format(num_components))
                continue

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_dic = float("-inf")

        num_categories = len(self.words)

        if num_categories == 1:
            raise ValueError("Length of words must be greater than 1 to use SelectorDIC for model selection.")

        anti_hwords = {k:v for (k,v) in self.hwords.items() if k != self.this_word}

        for num_components in range(self.min_n_components, self.max_n_components + 1):
            candidate_model = self.base_model(num_components)

            try:
                likelihood = candidate_model.score(self.X, self.lengths)

                sum_anti_likliehoods = 0

                for word in anti_hwords:
                    X, lengths = self.hwords[word]
                    sum_anti_likliehoods += candidate_model.score(X, lengths)

                average_anti_likliehood = sum_anti_likliehoods / (num_categories - 1)
                dic = likelihood - average_anti_likliehood

                if dic > best_dic:
                    best_model = candidate_model
                    best_dic = dic

            except ValueError:
                if self.verbose:
                    print("failure to calculate DIC with {} components".format(num_components))
                continue

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_avg_logL = float("-inf")

        for num_components in range(self.min_n_components, self.max_n_components + 1):
            split_method = KFold(n_splits=min(3, len(self.sequences)))
            logLs = []

            try:
                candidate_model = GaussianHMM(n_components=num_components, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False)
            except ValueError:
                continue

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                try:
                    candidate_model.fit(X_train, lengths_train)
                    logL = candidate_model.score(X_test, lengths_test)
                    logLs.append(logL)

                except ValueError:
                    continue

            if logLs:
                avg_logL = np.mean(logLs)

                if avg_logL > best_avg_logL:
                    best_model = candidate_model
                    best_avg_logL = avg_logL

        return best_model
