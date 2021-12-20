import numpy as np
import pandas as pd

# depent on tensorflow 1.14
import tensorflow as tf
from mia.estimators import ShadowModelBundle, prepare_attack_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1
# privacy package
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer

from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy

import sklearn.metrics

# set random seed
np.random.seed(19122)

GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer



class DataGenerator(object):
    """
    Load and preprocess data: 
    """

    def __init__(self, csv_path):
        super(DataGenerator, self).__init__()

        self.csv_path = csv_path 

        model_input = pd.read_csv(self.csv_path)
        print ('Base data has %i rows and %i columns' % (model_input.shape[0], model_input.shape[1]))      

        row_count = model_input.shape[0]
        patient_count = len(model_input['patient_id'].unique())
        if row_count == patient_count:
            print ('Row Count: ', model_input.shape[0])
            print ('Patient Count: ', len(model_input['patient_id'].unique()))
        else:
            raise ValueError('Model Input File is not at required level of data (patient_id)')

        model_input.groupby(['cohort_type','cohort_flag']).patient_id.nunique()

        model_input = model_input.drop(['patient_id','cohort_type'], axis = 1)

        target_map = {u'1': 1, u'0': 0}
        model_input['__target__'] = model_input['cohort_flag'].map(str).map(target_map)
        model_input = model_input.drop(['cohort_flag'], axis = 1)

        model_input.groupby(['__target__']).count()

        potential_target_leaks = ['cardiomyopathy_in_diseases_classified_elsewhere','other_forms_of_heart_disease']

        model_input_flt_leaks = model_input.drop(potential_target_leaks, axis = 1)


        self.X = model_input_flt_leaks.drop('__target__', axis=1)
        # self.y = np.array(model_input_flt_leaks['__target__'])
        self.y = model_input_flt_leaks['__target__']




def split_to_be_divisible(X, y, shadow_perc, batch_size):
    """
    Split a dataframe into target dataset and shadow dataset, and make them divisible by batch size.

    :param X: genotype data
    :param y: phenotype data
    :param shadow_perc: specified percent for shadow dataset, target_perc = 1 - shadow_perc
    :param batch_size: batch_size for training process

    :return: target datasets, shadow datasets
    """

    # stop and output error, if X and y have different number of individuals.
    assert y.shape[0] == X.shape[0]

    # calculate sample size of target and shadow
    total_row = X.shape[0]
    num_shadow_row = int(total_row * shadow_perc) - int(total_row * shadow_perc) % batch_size
    num_target_row = (total_row - num_shadow_row) - (total_row - num_shadow_row) % batch_size

    # split train and valid
    random_row = np.random.permutation(total_row)
    shadow_row = random_row[:num_shadow_row]
    target_row = random_row[-num_target_row:]

    target_X = X.iloc[target_row]
    shadow_X = X.iloc[shadow_row]

    target_y = y.iloc[target_row]
    shadow_y = y.iloc[shadow_row]

    return target_X, target_y, shadow_X, shadow_y


def target_model():
    """The architecture of the target model.
    The attack is white-box, hence the attacker is assumed to know this architecture too.

    :return: target model
    """

    classifier = Sequential()
    classifier.add(
        Dense(1,
              input_dim=feature_size,
              kernel_regularizer=l1(kernel_regularization),
              activation='sigmoid'))

    if dpsgd:
        optimizer = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=int(microbatches_perc * batch_size),
            learning_rate=learning_rate)
        # Compute vector of per-example loss rather than its mean over a minibatch.
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.compat.v2.losses.Reduction.NONE)
    else:
        optimizer = GradientDescentOptimizer(learning_rate=learning_rate)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    classifier.summary()

    # Compile model with Keras
    classifier.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return classifier


def shadow_model():
    """The architecture of the shadow model is same as target model, because the attack is white-box,
    hence the attacker is assumed to know this architecture too.

    :return: shadow model
    """

    classifier = Sequential()
    classifier.add(
        Dense(1,
              input_dim=feature_size,
              kernel_regularizer=l1(kernel_regularization),
              activation='sigmoid'))

    if dpsgd:
    # if 0:
        optimizer = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=int(microbatches_perc * batch_size),
            learning_rate=learning_rate)
        # Compute vector of per-example loss rather than its mean over a minibatch.
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.compat.v2.losses.Reduction.NONE)
    else:
        optimizer = GradientDescentOptimizer(learning_rate=learning_rate)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Compile model with Keras
    classifier.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return classifier


def main():
    print("Training the target model...")
    # split target dataset to train and valid, and make them evenly divisible by batch size
    target_X_train, target_y_train, target_X_valid, target_y_valid = split_to_be_divisible(target_X,
                                                                                           target_y,
                                                                                           0.2,
                                                                                           batch_size)

    print('>>>>>>>>>>> main() shapes  ',target_X_train.shape, target_y_train.shape, target_X_valid.shape)

    tm = target_model()
    tm.fit(target_X_train.values,
           target_y_train.values,
           batch_size=batch_size,
           epochs=epochs,
           validation_data=[target_X_valid.values, target_y_valid.values],
           verbose=verbose)

    outs = tm.predict(unused_X)
    acc = sklearn.metrics.accuracy_score(unused_y,  np.round(np.squeeze(outs)))
    print('target accuracy on unused: ', acc)
    target_accs.append(acc)
    print('______ ')


    print("Training the shadow models.")
    # train only one shadow model
    SHADOW_DATASET_SIZE = int(shadow_X.shape[0] / 2)
    smb = ShadowModelBundle(
        shadow_model,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=1,
    )
    # Training the shadow models with same parameter of target model, and generate attack data...
    attacker_X, attacker_y = smb.fit_transform(shadow_X.values, shadow_y.values,
                                               fit_kwargs=dict(epochs=epochs,
                                                               batch_size=batch_size,
                                                               verbose=verbose),
                                               )
    smb_model = smb._get_model(0)
    outs = smb_model.predict(unused_X)
    acc = sklearn.metrics.accuracy_score(unused_y,  np.round(np.squeeze(outs)))
    print('shadow accuracy on unused: ', acc)
    shadow_accs.append(acc)
    print('______ ')


    print("Training attack model...")
    clf = RandomForestClassifier(max_depth=2)
    clf.fit(attacker_X, attacker_y)

    # Test the success of the attack.
    ATTACK_TEST_DATASET_SIZE = unused_X.shape[0]
    # Prepare examples that were in the training, and out of the training.
    data_in = target_X_train[:ATTACK_TEST_DATASET_SIZE], target_y_train[:ATTACK_TEST_DATASET_SIZE]
    data_out = unused_X[:ATTACK_TEST_DATASET_SIZE], unused_y[:ATTACK_TEST_DATASET_SIZE]

    # Compile them into the expected format for the AttackModelBundle.
    attack_test_data, real_membership_labels = prepare_attack_data(tm, data_in, data_out)

    print('>>>>>>>>>>> main() shapes  ', SHADOW_DATASET_SIZE, attacker_X.shape, ATTACK_TEST_DATASET_SIZE, attack_test_data.shape, real_membership_labels.shape)

    # Compute the attack accuracy.
    attack_guesses = clf.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)

    print('attack accuracy: {}'.format(attack_accuracy))
    attack_accs.append(attack_accuracy)
    print('______ ')


if __name__ == '__main__':

    ## needed for GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    csv_path = "IDASH2021-CH3-data.csv"

    # parameters
    dpsgd = True

    # target model hyper-parameters same as Lasso-dp
    epochs = 40
    batch_size = 16
    microbatches_perc = 1.0
    learning_rate = 0.015
    kernel_regularization = 0.0
    noise_multiplier = 1
    l2_norm_clip = 1.0
    verbose=0


    print("Loading and splitting dataset")
    the_data = DataGenerator(csv_path)
    
    print('>>>>>>>>>>> MAIN shapes the_data ',the_data.X.shape, the_data.y.shape)

    target_X, target_y, shadow_X, shadow_y = split_to_be_divisible(the_data.X,
                                                                   the_data.y,
                                                                   0.5,
                                                                   batch_size=80)
    
    print('>>>>>>>>>>> MAIN shapes target shadow unused ',target_X.shape, target_y.shape, shadow_X.shape)

    shadow_X, shadow_y, unused_X, unused_y = split_to_be_divisible(shadow_X,
                                                                   shadow_y,
                                                                   0.2,
                                                                   batch_size)

    feature_size = target_X.shape[1]

    print('>>>>>>>>>>> MAIN shapes target shadow unused ',target_X.shape, target_y.shape, shadow_X.shape,  unused_X.shape, feature_size)


    target_accs = []
    shadow_accs = []
    attack_accs = []
    epsilons = []

    N = target_X.shape[0]
    delta = 1/N

    epochs = 40
    runs = 4    

    noise_multipliers = np.arange(0.5, 4.5, 0.25)
    noise_multipliers = np.append(noise_multipliers, [5, 10, 15, 20, 30, 50])
    
    #######  temporarily disable dp to calculate acc without DP
    if 0: 
        dpsgd = False
        noise_multipliers = [-1] # does not matter if no dp
    #######

    for noise_multiplier in noise_multipliers:
        eps, _ = compute_dp_sgd_privacy(N, batch_size, noise_multiplier, epochs, delta)         ##  get eps for given params and delta
        print(noise_multiplier, eps)
        epsilons.append(eps)
        for i in range(runs):
            main()
            print('')

    print(noise_multipliers)
    
    arr = np.asarray(target_accs)
    target_accs = np.mean(arr.reshape(-1, runs), axis=1)
    arr = np.asarray(shadow_accs)
    shadow_accs = np.mean(arr.reshape(-1, runs), axis=1)
    arr = np.asarray(attack_accs)
    attack_accs = np.mean(arr.reshape(-1, runs), axis=1)

    print(target_accs)
    print(shadow_accs)
    print(attack_accs)
    print(epsilons)

    