from .features import FeatureMap, Statistic
from .cell import Dataset, Cell

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from glmnet import LogitNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from glmnet.scorer import make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss, f1_score, matthews_corrcoef


from imblearn.over_sampling import SMOTE
from itertools import combinations

import copy
import pickle
import json
import datajoint as dj
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


schema = dj.schema('agberens_morphologies', locals())

@schema
class Classifier(dj.Lookup):
    definition = """
    classifier_id: serial # 
    ---
    classifier: varchar(100) # type of the classifier used. E.g. kNN = k-Nearest Neighbor or Logit = Logistic Regression
    params: varchar(200) # json dumped dictionary of classifier parameters. Make sure the names fit with function description
    ___
    UNIQUE INDEX(classifier, params)
    """

    def get_classifier(self, key):

        assert len(Classifier() & key) == 1, "Please select only one classifier"
        classifier = (Classifier() & key).fetch1('classifier')
        classifier_params = json.loads((Classifier() & key).fetch1('params'))

        # create classifier
        if classifier == 'kNN':
            m = KNeighborsClassifier(**classifier_params)
        elif classifier == 'Logit':
            m = LogitNet(random_state=17, **classifier_params)
            m.scoring = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
        elif classifier == 'randomForest':
            m = RandomForestClassifier(**classifier_params)
        elif classifier == 'SVC':
            m = SVC(**classifier_params)
        elif classifier == 'DecisionTree':
            m = DecisionTreeClassifier(**classifier_params)
        else:
            raise NotImplementedError('There is no classifier implemented for {0}'.format(classifier))
        return m


@schema
class ClassificationPair(dj.Computed):
    definition = """
    # only cell types including more than 5 cells are included in classification analysis
    -> Dataset
    group_a : varchar(25)
    group_b : varchar(25)
    ---

    UNIQUE INDEX(ds_id, group_a, group_b)
    """

    def _make_tuples(self, key):
        cells = pd.DataFrame(((Cell() - Cell().Ignore()) & key).fetch(as_dict=True))

        # exclude cell types that have less than 5 cells
        types = np.unique(cells['type'])[(cells.groupby(['ds_id', 'type']).count() > 5).values.reshape(-1)]
        pairs = list(combinations(types, 2))
        self.insert([(key['ds_id'], p[0], p[1]) for p in pairs])


def get_pca_transformed_data(X_trn, X_tst, pca, fm_lengths, var=.9, scaling=True):
    indices = np.cumsum(fm_lengths)

    X_train_ = np.zeros((X_trn.shape))
    X_test_ = np.zeros((X_tst.shape))
    std_dev = np.ones(X_trn.shape[1])

    # how many single value stats are included?
    u, c = np.unique(fm_lengths, return_counts=True)

    # if PCA has n_components specified
    pca_n = pca.n_components
    if pca_n:
        output_dims = np.array([k if k <= pca_n else min(pca_n, X_trn.shape[0]) for k in fm_lengths])

        for k in range(1, fm_lengths.shape[0]):
            start = indices[k - 1]
            stop = indices[k]

            o_start = np.cumsum(output_dims)[k - 1]
            o_stop = np.cumsum(output_dims)[k]

            if fm_lengths[k] > pca_n:  # perform PCA when the length of the feature map exceeds number of components specified
                pca_X = pca.fit_transform(X_trn[:, start:stop])
                X_train_[:, o_start:o_stop] = pca_X
                X_test_[:, o_start:o_stop] = pca.transform(X_tst[:, start:stop])

                if scaling:
                    # scale by std of first component for when the features are combined
                    X_train_[:, o_start:o_stop] /= np.std(pca_X[:, 0])
                    X_test_[:, o_start:o_stop] /= np.std(pca_X[:, 0])

            else:
                if fm_lengths[k] == 1 and c[u == 1] > 1:
                    std_dev[o_start:o_stop] = get_std_dev(X_trn[:, start:stop])
                X_train_[:, o_start:o_stop] = (X_trn[:, start:stop] - np.mean(X_trn[:, start:stop]))
                X_test_[:, o_start:o_stop] = (X_tst[:, start:stop] - np.mean(X_tst[:, start:stop]))
    elif var:
        # reduce according to variance kept

        o_start = 0
        for k in range(1, fm_lengths.shape[0]):
            start = indices[k - 1]
            stop = indices[k]

            if stop - start > 1:  # it's not a morphometric
                pca_X = pca.fit_transform(X_trn[:, start:stop])
                idx = np.argmax(np.cumsum(pca.explained_variance_ratio_) > var) + 1

                X_train_[:, o_start:o_start + idx] = pca_X[:, :idx]
                X_test_[:, o_start:o_start + idx] = pca.transform(X_tst[:, start:stop])[:, :idx]
                if scaling:
                    # scale by std of first component for when the features are combined
                    X_train_[:, o_start:o_start + idx] /= np.std(pca_X[:, 0])
                    X_test_[:, o_start:o_start + idx] /= np.std(pca_X[:, 0])

            else:
                idx = 1
                if c[u == 1] > 1:
                    std_dev[o_start:o_start + idx] = get_std_dev(X_trn[:, start:stop])
                X_train_[:, o_start:o_start + idx] = (X_trn[:, start:stop] - np.mean(X_trn[:, start:stop]))
                X_test_[:, o_start:o_start + idx] = (X_tst[:, start:stop] - np.mean(X_tst[:, start:stop]))

            o_start += idx
        o_stop = copy.copy(o_start)

    # perform z-scoring. If no morphometrics were involved the std_dev only contains ones
    X_train = copy.copy(X_train_[:, :o_stop]) / std_dev[:o_stop]
    X_test = copy.copy(X_test_[:, :o_stop]) / std_dev[:o_stop]

    if len(X_train.shape) > 2:
        X_train.squeeze(), X_test.squeeze()
    return X_train, X_test


def get_std_dev(obs):
    std_dev = np.std(obs, axis=0)
    zero_std_mask = std_dev == 0
    if zero_std_mask.any():
        std_dev[zero_std_mask] = 1.0
    return std_dev


def get_fm_lengths(z):
    combined_feature = ((Statistic() & z).fetch('statistic_type') == 'combined')
    if z['part_id'] == 4 and combined_feature:
        fm_lengths = []
        for i in (Statistic().Combined() & z).fetch('involved_ids')[0]:
                fm_lengths += (FeatureMap()).get_dimension_of_feature(dict(statistic_id=i, part_id=1))
        fm_lengths += fm_lengths #duplicate it
    elif combined_feature:
        fm_lengths = []
        for i in (Statistic().Combined() & z).fetch('involved_ids')[0]:
            fm_lengths += (FeatureMap()).get_dimension_of_feature(dict(statistic_id=i, part_id=z['part_id']))

    else:
        fm_lengths = FeatureMap().get_dimension_of_feature(z)

    fm_lengths.insert(0, 0)
    fm_lengths = np.array(fm_lengths)
    return fm_lengths


def _get_cross_validation_runs(X,y,m,kf,pca,fm_lengths,key):
    runs = []
    k = 1

    for train_ix, test_ix in kf.split(X, y):

        X_train = X[train_ix]
        y_train = y[train_ix]
        X_test = X[test_ix]
        y_test = y[test_ix]

        X_train, X_test = get_pca_transformed_data(X_train, X_test, pca, fm_lengths)

        r_ = copy.copy(key)
        r_['run_id'] = k

        try:
            m.fit(X_train, y_train)

            r_['accuracy_train'] = m.score(X_train, y_train)
            r_['accuracy_test'] = m.score(X_test, y_test)

            r_['f1_train'] = f1_score(y_train, m.predict(X_train), average='macro')
            r_['f1_test'] = f1_score(y_test, m.predict(X_test), average='macro')

            r_['mcc_train'] = matthews_corrcoef(y_train,m.predict(X_train))
            r_['mcc_test'] = matthews_corrcoef(y_test,m.predict(X_test))
            
            r_['log_loss_train'] = log_loss(y_train, m.predict_proba(X_train), labels=np.unique(y_train))
            r_['log_loss_test'] = log_loss(y_test, m.predict_proba(X_test), labels=np.unique(y_test))

            r_['test_indices'] = test_ix
            r_['training_indices'] = train_ix
            runs.append(r_)
            k += 1
        except (ValueError, RuntimeError) as e:
            print(e)
            continue
    return runs


def plot_accuracy_matrix(table, score, key, vmin=.5, vmax=1, cmap='YlGn', **kwargs):

    if table == 'balanced_pairwise':
        t = PairwiseClassificationBalanced()
    elif table == 'imbalanced_pairwise':
        t = PairwiseClassificationImbalanced()
    elif table == 'balanced_multi':
        t = MultiClassClassificationBalanced()
    elif table == 'imbalanced_multi':
        t = MultiClassClassificationImbalanced()
    else:
        raise ValueError("table %s is not defined!"%table)

    if score == 'accuracy':
        s = 'accuracy_test'
    elif score == 'log_loss':
        s = 'log_loss_test'
    elif score == 'mcc':
        s = 'mcc_test'
    elif score =='f1':
        s = 'f1_test'
    
    d = pd.DataFrame((t.CVRun() & key).fetch(as_dict=True))
    del d['training_indices']
    del d['test_indices']

    grouped = d.groupby(
        ['ds_id', 'statistic_id', 'part_id', 'reduction_id', 'classifier_id', 'group_a', 'group_b']).mean()

    # create pivot table of avg performance
    p = grouped.pivot_table(index='group_a', columns='group_b', values=s, fill_value=0)
    index = p.index.union(p.columns)
    p = p.reindex(index=index, columns=index, fill_value=0)

    mask = np.zeros_like(p, dtype=np.bool)
    mask[np.tril_indices_from(mask)] = True

    return sns.heatmap(p, mask=mask, cmap=cmap, vmin=vmin, vmax=vmax, center=(vmax + vmin)/2., square=True, linewidths=.5,
                       cbar_kws={"shrink": .5}, **kwargs)


def compare_performance(table, score, key1, key2):
    
    if table == 'balanced_pairwise':
        t = PairwiseClassificationBalanced()
    elif table == 'balanced_multi':
        t = MultiClassClassificationBalanced()
    elif table == 'imbalanced_pairwise':
        t = PairwiseClassificationImbalanced()
    elif table == 'imbalanced_multi':
        t = MultiClassClassificationImbalanced()
    else:
        raise ValueError("table %s is not defined!" % table)

    if score == 'accuracy':
        s = 'accuracy_test'
    elif score == 'log_loss':
        s = 'log_loss_test'
    elif score == 'brier':
        s = 'brier_score_test'
    elif score =='f1':
        s = 'f1_test'

    else:
        raise ValueError("score %s is not defined!" % score)

    assert (len(t & key1) == len(t & key2)), \
        "key1 and key2 should identify corresponding tuples"

    assert (key1['ds_id'] == key2['ds_id']), "Performance of statistic %s and statistic %s can only be compared " \
                                             "within the SAME data set." \
                                             % (key1['statistic_id'], key2['statistic_id'])

    scores1 = get_scores(table, key1)
    scores2 = get_scores(table, key2)

    m = np.round(np.min((np.min(scores1[s]), np.min(scores2[s]))), 1)

    plt.plot([min(.4, m), 1.], [min(.4, m), 1.])
    sns.despine()
    plt.scatter(scores1[s], scores2[s], c='k', alpha=0.3, label='pair performance')

    indices = np.where(np.array(list(key1.values())) != np.array(list(key2.values())))[0]
    xlabel = ''
    ylabel = ''
    for idx in indices:
        xlabel = xlabel + list(key1.keys())[idx] + " " + str(list(key1.values())[idx]) + "\n"
        ylabel = ylabel + list(key2.keys())[idx] + " " + str(list(key2.values())[idx]) + "\n"

    xlabel += '[avg test performance [%s]]' % score
    ylabel += '[avg test performance [%s]' % score

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


#### Classification using other scores and no SMOTE ###

@schema
class PairwiseClassificationImbalanced(dj.Computed):
    definition = """
    # stores the model that has been fitted for pairwise classification
    -> FeatureMap
    -> Classifier
    -> ClassificationPair
    ---
    model  : longblob          # pickled model
    """

    @property
    def key_source(self):
        return (FeatureMap() - FeatureMap().Ignore())*Classifier()*ClassificationPair()

    def _make_tuples(self, key):

        print('Populating key ', key)
        z = copy.copy(key)
        m = Classifier().get_classifier(z)

        # get data
        df = FeatureMap().get_as_dataframe(z, dj.OrList(("type='%s'" % z['group_a'], "type='%s'" % z['group_b'])))
        X = df.replace(np.inf, np.nan).fillna(0).values.T
        y = df.columns.get_level_values('type')

        # if the feature map is a combined one get the indices of each fm to do pca separately
        fm_lengths = get_fm_lengths(z)

        pca = PCA(copy=True, whiten=False)
        kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=17)

        runs = _get_cross_validation_runs(X, y, m, kf, pca, fm_lengths,z)

        X, _ = get_pca_transformed_data(X, X, pca, fm_lengths)

        try:
            m.fit(X, y)
            b = pickle.dumps(m)
            z['model'] = b.hex()

            # insert key
            self.insert1(z)

            self.CVRun().insert(runs)
        except (ValueError, RuntimeError) as e:
            print(e)
            print('model was neither fit nor stored.')

    class CVRun(dj.Part):
        definition = """
         -> master
         run_id: int
         ---
         training_indices: blob   # indices of training data
         test_indices: blob       # indices of test data
         accuracy_train: float      # accuracy score of the model on training set
         accuracy_test: float       # accuracy score of the model on test set
         f1_train:      float       # Macro F1 score of the model on the training set
         f1_test:       float       # Macro F1 score of the model on the test set
         mcc_train: float           #  Matthew correlation coefficient on the training set
         mcc_test: float            # Matthew correlation coefficient on the test set
         log_loss_train: float   #  log loss on training set
         log_loss_test: float       # log loss on test set
         """


@schema
class MultiClassClassificationImbalanced(dj.Computed):
    definition = """
    -> Dataset
    -> FeatureMap
    -> Classifier
    ---
    model : longblob # pickled model fitted on all data
    no_classes: int  # number of classes in data
    """

    class CVRun(dj.Part):
        definition = """
        -> master
        run_id               : smallint
        ---
        training_indices     : blob                         # indices of training data
        test_indices         : blob                         # indices of test data
        accuracy_train: float                               # accuracy score of the model of training set
        accuracy_test: float                                # accuracy score of the model of test set
        f1_train:      float       # Macro F1 score of the model on the training set
        f1_test:       float       # Macro F1 score of the model on the test set
        mcc_train: float           #  Matthew correlation coefficient on the training set
        mcc_test: float            # Matthew correlation coefficient on the test set
        log_loss_train: float                               # log loss of training set
        log_loss_test: float                                # log loss of test set 
        """

    @property
    def key_source(self):
        return (FeatureMap() - FeatureMap().Ignore())*Dataset()*Classifier()

    def _make_tuples(self, key):

        print('Populating key ', key)
        z = copy.copy(key)
        # get classifier
        m = Classifier().get_classifier(z)

        # get data
        df = FeatureMap().get_as_dataframe(z)
        X = df.replace(np.inf, np.nan).fillna(0).values.T
        y = df.columns.get_level_values('type')

        # get rid of data that has less than 5 data points and that is a pyramidal cell
        # TODO Here is the pyramidal cell exception try to solve it more elegantly?
        u, c = np.unique(y, return_counts=True)
        idx = np.logical_and([k in u[c > 5] for k in y], [k != 'pyr' for k in y])
        X = X[idx]
        y = y[idx]

        # if the feature map is a combined one get the indices of each fm to do pca separately
        fm_lengths = get_fm_lengths(z)

        pca = PCA(copy=True, whiten=False)
        kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=17)

        runs = _get_cross_validation_runs(X, y, m, kf, pca, fm_lengths, z)

        # fit model on entire data to get coefficient vectors

        X, _ = get_pca_transformed_data(X, X, pca, fm_lengths)

        try:
            m.fit(X, y)
            b = pickle.dumps(m)
            z['model'] = b.hex()

            z['no_classes'] = np.unique(y).shape[0]
            # insert key
            self.insert1(z)

            self.CVRun().insert(runs)
        except (ValueError, RuntimeError) as e:
            print(e)
            print('model was neither fit nor stored.')


def get_model(table, key):
    
    if table == 'imbalanced_pairwise':
        hex_models = (PairwiseClassificationImbalanced() & key).fetch('model')

    elif table == 'imbalanced_multi':
        hex_models = (MultiClassClassificationImbalanced() & key).fetch('model')
    else:
        raise ValueError("table %s is not defined!"%table)
    
    models = []
    for h in hex_models:
        m = pickle.loads(bytes.fromhex(h[0]))
        models.append(m)

    return tuple(models)


def get_scores(table, key):
    
    if table == 'imbalanced_pairwise':
        d = (PairwiseClassificationImbalanced().CVRun() & key).fetch(as_dict=True)
        groupby_ids =  ['statistic_id', 'ds_id', 'part_id', 'reduction_id', 'classifier_id', 'group_a', 'group_b']
    elif table == 'imbalanced_multi':
        d = (MultiClassClassificationImbalanced().CVRun() & key).fetch(as_dict=True)
        groupby_ids =  ['statistic_id', 'ds_id', 'part_id', 'reduction_id', 'classifier_id']
    else:
        raise ValueError("table %s is not defined!"%table)

    for entry in d:
        del entry['test_indices']
        del entry['training_indices']

    df_ = pd.DataFrame(d)
    df_mean = df_.groupby(groupby_ids).mean()

    try:
        del df_mean['run_id']
    except KeyError as e:
        print('Warning: ', e)  

    df = df_mean.reset_index()
    return df

 