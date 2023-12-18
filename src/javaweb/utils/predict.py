import numpy as np
import pandas as pd
import joblib
import torch


import src.CMAPSS.CMAPSSCNN as CMAPSS_CNN

data_dir = 'data/java/'
model_dir = 'models/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_data_rul(file_path, kmeans_path, scaler_path):
    data = pd.read_csv(file_path, index_col=0)
    data.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
              's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
              's18', 's19', 's20', 's21']
    data = data.sort_values(by='cycle')
    # print(data)
    data = cluster(kmeans_path, data)
    # print(data)
    data = oc_cols(data)
    data_df, cols_wo_na = get_scaling(data, scaler_path)
    # print(data)
    data.loc[:, cols_wo_na] = data_df.loc[:, cols_wo_na]
    data_sequence = sequence_generator(data, cols_wo_na)
    return data_sequence
    pass


def cluster(kmeans_path, data):
    estimator = joblib.load(kmeans_path)
    input_kmeans = data[["setting1", "setting2", "setting3"]]
    data_labels = estimator.predict(input_kmeans)
    print(data_labels)
    data["operating_condition"] = data_labels
    return data


def oc_cols(data):
    if 'operating_condition' not in data.columns:
        print('Please check operating condition is in file')
    else:
        data[["oc_0", "oc_1", "oc_2", "oc_3", "oc_4", "oc_5"]] = pd.DataFrame([[0, 0, 0, 0, 0, 0]],
                                                                              index=data.index)
        group_by = data.groupby('id', sort=False)
        additional_oc = []
        for _, data_it in group_by:
            data_it = data_it.reset_index(drop=True)
            for i in range(1, data_it.shape[0]):
                check_oc = data_it.at[i - 1, 'operating_condition']
                update_cols = ['oc_' + str(int(check_oc))]
                data.loc[i:, update_cols] = data.loc[i:, update_cols] + 1

            additional_oc.append(data)

        oc_cols_ = pd.concat(additional_oc, sort=False, ignore_index=True)
        # oc_cols_ = pd.DataFrame(oc_cols_, index=None)
        # print(oc_cols_)
        return oc_cols_


def get_scaling(data, scaler_path):
    sensors = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
               's15', 's16', 's17', 's18', 's19', 's20', 's21']
    scaler = joblib.load(scaler_path)
    # print(scaler)
    group_oc = data.groupby('operating_condition', sort=False)
    scaled_sensor = []
    for operating_condition, data_it in group_oc:
        if scaler['scaler'] == 'mm':
            min_ = scaler['min_oc' + str(operating_condition)]
            max_ = scaler['max_oc' + str(operating_condition)]
            scaler_range = scaler['scaler_range']
            scaled_data = (((data_it[sensors] - min_) / (max_ - min_)) * (
                    scaler_range[1] - scaler_range[0])) + scaler_range[0]

        scaled_sensor.append(scaled_data)

    scaled_df = pd.concat(scaled_sensor, sort=False)
    scaled_df = scaled_df.sort_index(axis=0, ascending=True)

    scaled_df = scaled_df.dropna(axis=1)
    cols_wo_na = scaled_df.columns

    # print(scaled_df)
    # print(cols_wo_na)

    return scaled_df, cols_wo_na


def sequence_generator(data, cols_wo_na, window_size=30):
    input_features = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20",
                      "s21"]
    group_by_id = data.groupby('id', sort=False)
    sequence = []
    for id, data_it in group_by_id:
        input_data = data_it[input_features]
        for p in range(window_size, input_data.shape[0] + 1):
            x = input_data.iloc[p - window_size: p, 0:]
            x = x.to_numpy()
            sequence.append(x)

    # print(sequence)
    sequence = torch.FloatTensor(sequence)
    sequence = sequence.view(sequence.shape[0], 1, window_size, len(input_features))
    return sequence


def predict_rul(data, model_path):
    model = CMAPSS_CNN.CNN()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    data_device = data.to(device)
    print(model)
    print(data.shape)
    model.eval()
    output = model(data_device.float())
    output = output.detach().cpu().numpy()
    output = output.flatten()
    rul = output[0]
    print('rul: ', rul)
    return rul


# if __name__ == '__main__':
#     data = load_data_rul('../../../data/java/rul_test.csv', '../../../models/kmeans_model.joblib',
#                          '../../../models/scaler_params.pkl')
#     print(predict_rul(data, '../../CMAPSS/results/202312181709_001.pth'))
