import math

import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from skimage.measure.entropy import shannon_entropy as sentropy
from tqdm import tqdm_notebook

def resample_augment(x_data, org_x, y_data, sample_flag, bf0=0, bf1=1, bf2=2, step_size=2):
    # Resample parameters
    ef0 = x_data.shape[2] - 3
    ef1 = x_data.shape[2] - 2
    ef2 = x_data.shape[2] - 1

    # Resample/augment
    x_data_a = x_data[:, bf0:ef0:step_size, bf1:ef1:step_size, :, :]
    x_data_b = x_data[:, bf1:ef1:step_size, bf1:ef1:step_size, :, :]
    x_data_c = x_data[:, bf2:ef2:step_size, bf1:ef1:step_size, :, :]
    x_data_d = x_data[:, bf0:ef0:step_size, bf2:ef2:step_size, :, :]
    x_data_e = x_data[:, bf1:ef1:step_size, bf2:ef2:step_size, :, :]
    x_data_f = x_data[:, bf2:ef2:step_size, bf2:ef2:step_size, :, :]
    x_data_g = x_data[:, bf0:ef0:step_size, bf0:ef0:step_size, :, :]
    x_data_h = x_data[:, bf1:ef1:step_size, bf0:ef0:step_size, :, :]
    x_data_i = x_data[:, bf2:ef2:step_size, bf0:ef0:step_size, :, :]

    if sample_flag == 'all':
        x_data = np.concatenate([x_data_b, x_data_c, x_data_f], axis=0)
        y_data = np.concatenate([y_data, y_data, y_data])
    elif sample_flag == 'a':
        x_data = np.concatenate([x_data_a], axis=0)
    elif sample_flag == 'b':
        x_data = np.concatenate([x_data_b], axis=0)
    elif sample_flag == 'c':
        x_data = np.concatenate([x_data_c], axis=0)
    elif sample_flag == 'd':
        x_data = np.concatenate([x_data_d], axis=0)
    elif sample_flag == 'e':
        x_data = np.concatenate([x_data_e], axis=0)
    elif sample_flag == 'f':
        x_data = np.concatenate([x_data_f], axis=0)
    elif sample_flag == 'g':
        x_data = np.concatenate([x_data_g], axis=0)
    elif sample_flag == 'h':
        x_data = np.concatenate([x_data_h], axis=0)
    elif sample_flag == 'i':
        x_data = np.concatenate([x_data_i], axis=0)
    elif sample_flag == 'none':
        pass

    org_x = np.array(0)

    if org_x.shape == ():
        mean_all_x = x_data.mean()
        x_data = x_data - mean_all_x
    else:
        mean_all_x = org_x.mean()
        x_data = x_data - mean_all_x

    return x_data.astype('float16'), y_data


def data_norm_stand(x_data):
    x_max = x_data.max(axis=(1, 2, 3))
    x_max = x_max[:, np.newaxis, np.newaxis, np.newaxis, :]
    x_std = x_data.astype('float64').std(axis=(1, 2, 3))
    x_std = x_std[:, np.newaxis, np.newaxis, np.newaxis, :]
    x_mean = x_data.mean(axis=(1, 2, 3))
    x_mean = x_mean[:, np.newaxis, np.newaxis, np.newaxis, :]
    x_data = 8192 * (x_data - x_mean) / x_std / x_max
    return x_data.astype('float16')

def filter_data(x_data, y_data, psa, mask_thresh=10.88):
    thresh_se_x1 = np.zeros(shape=x_data.shape[0], dtype='O')
    for i in range(x_data.shape[0]):
        thresh_se_x1[i] = sentropy(x_data[i].astype('float16'), 2)
    mask = thresh_se_x1 > mask_thresh
    x_data = x_data[mask]
    y_data = y_data[mask]
    psa = psa[mask]
    return x_data, y_data, psa

def create_patient_id(df):
    all_names = ['%s_%s' % (df['Hospital'][i], df['Patient'][i]) for i in range(len(df))]
    df['Name'] = all_names
    return df

def create_patient_label(df):
    all_labels = [0] * len(df)
    for i in range(len(df)):
        name = df['Name'][i]
        labels = df[df.Name == name].Label.tolist()
        if any(gs in label for gs in ['GS7', 'GS8', 'GS9', 'GS10'] for label in labels):
            all_labels[i] = 1
        else:
            all_labels[i] = 0
    
    df['CancerLabel'] = all_labels
    return df

def patient_stratify(df, test_ratio, shuffle_indexes):
    if not any("Name" in s for s in list(df.columns.values)):
        df = create_patient_id(df)
    
    all_names = list(dict.fromkeys(df['Name'].tolist()))
    
    if np.max(shuffle_indexes) == 0:
        shuffle_indexes = np.random.permutation(len(all_names))
        
    test_size = int(len(all_names) * test_ratio)
    
    test_names = [all_names[i] for i in shuffle_indexes[:test_size]]
    train_names = [all_names[i] for i in shuffle_indexes[test_size:]]
    
    train_set_df, test_set_df = pd.DataFrame(), pd.DataFrame()
    
    for name in train_names:
        train_set_patient = df[df['Name'] == name]
        train_set_df = pd.concat([train_set_df, train_set_patient])
    
    for name in test_names:
        test_set_patient = df[df['Name'] == name]
        test_set_df = pd.concat([test_set_df, test_set_patient])
    
    return train_set_df, test_set_df, shuffle_indexes


def train_test_split_by_name(df, test_ratio):
    if not any("Name" in s for s in list(df.columns.values)):
        df = create_patient_id(df)
    
    all_names = list(dict.fromkeys(df['Name'].tolist()))
    shuffle_indexes = np.random.permutation(len(all_names))
    test_size = int(len(all_names) * test_ratio)
    
    test_names = [all_names[i] for i in shuffle_indexes[:test_size]]
    train_names = [all_names[i] for i in shuffle_indexes[test_size:]]
    
    train_set_df, test_set_df = pd.DataFrame(), pd.DataFrame()
    
    for name in train_names:
        train_set_patient = df[df['Name'] == name]
        train_set_df = pd.concat([train_set_df, train_set_patient], ignore_index=True)
    
    for name in test_names:
        test_set_patient = df[df['Name'] == name]
        test_set_df = pd.concat([test_set_df, test_set_patient], ignore_index=True)
    
    return train_set_df, test_set_df


def cancer_stratify(df, ratio_num):
    if not any("Name" in s for s in list(df.columns.values)):
        df = create_patient_id(df)
    if not any("CancerLabel" in s for s in list(df.columns.values)):
        df = create_patient_label(df)
        
    cancer01 = df.loc[df['CancerLabel'] == 1].copy()
    cancer01 = cancer01.sample(frac=1).reset_index(drop=True)
    cancer02, can_test_df = train_test_split_by_name(cancer01, ratio_num)
    
    benign01 = df.loc[df['CancerLabel'] == 0].copy()
    benign01 = benign01.sample(frac=1).reset_index(drop=True)
    benign02, ben_test_df = train_test_split_by_name(benign01, ratio_num)
    
    temp_df02 = pd.concat([benign02, cancer02])
    test_df = pd.concat([ben_test_df, can_test_df])

    temp_df02 = temp_df02.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    
    return temp_df02, test_df

def hospital_stratify(df, ratio_num):
    uva_df01 = df.loc[df['Hospital'] == 'UVA'].copy()
    uva_df01 = uva_df01.sample(frac=1).reset_index(drop=True)
    uva_df02, uva_test_df = train_test_split_by_name(uva_df01, ratio_num)

    crceo_df01 = df.loc[df['Hospital'] == 'CRCEO'].copy()
    crceo_df01 = crceo_df01.sample(frac=1).reset_index(drop=True)
    crceo_df02, crceo_test_df = train_test_split_by_name(crceo_df01, ratio_num)

    pcc_df01 = df.loc[df['Hospital'] == 'PCC'].copy()
    pcc_df01 = pcc_df01.sample(frac=1).reset_index(drop=True)
    pcc_df02, pcc_test_df = train_test_split_by_name(pcc_df01, ratio_num)

    pmcc_df01 = df.loc[df['Hospital'] == 'PMCC'].copy()
    pmcc_df01 = pmcc_df01.sample(frac=1).reset_index(drop=True)
    pmcc_df02, pmcc_test_df = train_test_split_by_name(pmcc_df01, ratio_num)

    jh_df01 = df.loc[df['Hospital'] == 'JH'].copy()
    jh_df01 = jh_df01.sample(frac=1).reset_index(drop=True)
    jh_df02, jh_test_df = train_test_split_by_name(jh_df01, ratio_num)

    temp_df02 = pd.concat([uva_df02, crceo_df02, pcc_df02, pmcc_df02, jh_df02])
    test_df = pd.concat([uva_test_df, crceo_test_df, pcc_test_df, pmcc_test_df, jh_test_df])

    temp_df02 = temp_df02.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    
    return temp_df02, test_df

def encode_labels(y: np.ndarray) -> np.ndarray:
    labels = [2]*len(y) 
    for i in tqdm(range(0, len(y))):
        if y[i] == 'Benign':
            labels[i] = 0
        else:
            labels[i] = 1      
    encoded_labels = np.array(labels)
    return encoded_labels

def factor_int(n):
    val = math.ceil(math.sqrt(n))
    val2 = int(n/val)
    while val2 * val != float(n):
        val -= 1
        val2 = int(n/val)
    return val, val2

def create_4D(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, col_name: str, sub_gain: float):
    mat_size = np.squeeze(np.array(train_df[col_name][1])).shape
    
    x_train = np.zeros(shape=(len(train_df),mat_size[0],mat_size[1],mat_size[2]))
    y_train = np.zeros(shape=(len(train_df)),dtype='O')
    for i in range(len(train_df)):
        a = np.squeeze(np.array(train_df[col_name][i]))
        x_train[i,:,:,:] = a - sub_gain*int(train_df['Gain'][i]);
        y_train[i] = np.squeeze(np.array(train_df['Label'][i]))
        
    x_train = np.transpose(x_train, (0, 2, 3, 1))

    x_valid = np.zeros(shape=(len(valid_df),mat_size[0],mat_size[1],mat_size[2]))
    y_valid = np.zeros(shape=(len(valid_df)),dtype='O')
    for i in range(len(valid_df)):
        e = np.squeeze(np.array(valid_df[col_name][i]))
        x_valid[i,:,:,:] = e - sub_gain*int(valid_df['Gain'][i]);
        y_valid[i] = np.squeeze(np.array(valid_df['Label'][i]))

    x_valid = np.transpose(x_valid, (0, 2, 3, 1))

    x_test = np.zeros(shape=(len(test_df),mat_size[0],mat_size[1],mat_size[2]))
    y_test = np.zeros(shape=(len(test_df)),dtype='O')
    for i in range(len(test_df)):
        c = np.squeeze(np.array(test_df[col_name][i]))
        x_test[i,:,:,:] = c - sub_gain*int(test_df['Gain'][i])
        y_test[i] = np.squeeze(np.array(test_df['Label'][i]))

    x_test = np.transpose(x_test, (0, 2, 3, 1))
        
    # To Binary and Categorical
    y_train_b = encode_labels(y_train)
    y_valid_b = encode_labels(y_valid)
    y_test_b = encode_labels(y_test)

    y_train = to_categorical(y_train_b)
    y_valid = to_categorical(y_valid_b)
    y_test = to_categorical(y_test_b)

    train_n1, train_n2 = factor_int(x_train.shape[1])
    valid_n1, valid_n2 = factor_int(x_valid.shape[1])
    test_n1, test_n2 = factor_int(x_test.shape[1])
    
    # Reshape for use as 3D + chnl
    x_train_4D = x_train.reshape(x_train.shape[0],train_n1,train_n2,x_train.shape[2],x_train.shape[3])
    y_train_4D = y_train
    
    x_valid_4D = x_valid.reshape(x_valid.shape[0],valid_n1, valid_n2, x_valid.shape[2],x_valid.shape[3])
    y_valid_4D = y_valid

    x_test_4D = x_test.reshape(x_test.shape[0], test_n1, test_n2,x_test.shape[2],x_test.shape[3])
    y_test_4D = y_test
    
    weight_type = 'PctCancer'
    train_df[weight_type] = train_df[weight_type].astype('U32')
    train_df[weight_type] = train_df[weight_type].replace('[]', 0)
    pct_cancer = np.squeeze(np.array([train_df[weight_type]],dtype = 'float'))
    pct_cancer[np.where(np.logical_and(pct_cancer > 0, pct_cancer <= 25))] = 0.9
    pct_cancer[pct_cancer == 0] =  0.001
    pct_cancer[np.where(np.logical_and(pct_cancer > 25, pct_cancer <= 50))] = 1.2
    pct_cancer[np.where(np.logical_and(pct_cancer > 50, pct_cancer <= 75))] = 1.8
    pct_cancer[np.where(np.logical_and(pct_cancer > 75, pct_cancer <= 100))] = 1.8
    pct_cancer_weights = pct_cancer
    
    weight_type = 'PSA'
    train_df[weight_type] = train_df[weight_type].astype('U32')
    train_df[weight_type] = train_df[weight_type].replace('[]', 0)
    psa_train = np.squeeze(np.array([train_df[weight_type]],dtype = 'float'))
    psa_train = psa_train.reshape((psa_train.shape[0], 1))

    valid_df[weight_type] = valid_df[weight_type].astype('U32')
    valid_df[weight_type] = valid_df[weight_type].replace('[]', 0)
    psa_valid = np.squeeze(np.array([valid_df[weight_type]],dtype = 'float'))
    psa_valid = psa_valid.reshape((psa_valid.shape[0], 1))
    
    test_df[weight_type] = test_df[weight_type].astype('U32')
    test_df[weight_type] = test_df[weight_type].replace('[]', 0)
    psa_test = np.squeeze(np.array([test_df[weight_type]],dtype = 'float'))
    psa_test = psa_test.reshape((psa_test.shape[0], 1))
    
    return x_train_4D, y_train_4D, x_valid_4D, y_valid_4D, x_test_4D, y_test_4D, pct_cancer_weights, psa_train, psa_valid, psa_test

def ten_fold_names(df: pd.DataFrame) -> list:

    if any("Name" in s for s in list(df.columns.values)) == False:

        df = create_patient_id(df)  

    all_names = list(dict.fromkeys(df['Name'].tolist()))

    shuffle_indexes = np.random.permutation(len(all_names))

    all_names = np.array(all_names)[shuffle_indexes]

    Patients_per_fold = int(len(all_names)/10)
    n_fold = [0]*10

    n_fold[0] = all_names[:Patients_per_fold]
    n_fold[1] = all_names[Patients_per_fold + 1 : 2*Patients_per_fold]
    n_fold[2] = all_names[2*Patients_per_fold + 1 : 3*Patients_per_fold]
    n_fold[3] = all_names[3*Patients_per_fold + 1 : 4*Patients_per_fold]
    n_fold[4] = all_names[4*Patients_per_fold + 1 : 5*Patients_per_fold]
    n_fold[5] = all_names[5*Patients_per_fold + 1 : 6*Patients_per_fold]
    n_fold[6] = all_names[6*Patients_per_fold + 1 : 7*Patients_per_fold]
    n_fold[7] = all_names[7*Patients_per_fold + 1 : 8*Patients_per_fold]
    n_fold[8] = all_names[8*Patients_per_fold + 1 : 9*Patients_per_fold]
    n_fold[9] = all_names[9*Patients_per_fold + 1 :]

    return n_fold
    
def ten_fold_crosvalidation_train_test_gen(fold, df, n_fold):
    # fold is variable with names to use in a specific fold
    # n_fold

    train_names_ = n_fold
    train_names_ = np.delete(train_names_, fold)
    train_names = train_names_[0]
    for i in range(1,len(train_names_)):

        train_fold = np.array(train_names_[i])
        train_names = np.append([train_names], [train_fold])

    test_names = n_fold[fold]

    train_set_df, train_set_patient  = pd.DataFrame(), pd.DataFrame()
    test_set_df, test_set_patient  = pd.DataFrame(), pd.DataFrame()

    for i in tqdm_notebook(range(0,len(train_names))):

        train_set_patient = df[df['Name'] == train_names[i]]
        train_set_df = train_set_df.append(train_set_patient)
        train_set_df = train_set_df.sample(frac=1)
        train_set_df = train_set_df.sample(frac=1).reset_index(drop=True)

    for i in tqdm_notebook(range(0,len(test_names))):

        test_set_patient = df[df['Name'] == test_names[i]]
        test_set_df = test_set_df.append(test_set_patient)
        test_set_df = test_set_df.append(test_set_patient)
        test_set_df = test_set_df.sample(frac=1)
        test_set_df = test_set_df.sample(frac=1).reset_index(drop=True)

    return train_set_df, test_set_df