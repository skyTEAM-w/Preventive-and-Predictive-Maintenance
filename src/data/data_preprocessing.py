import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.decomposition import PCA


class preprocessor:
    def __init__(self, data=pd.DataFrame, data_col=list):
        self.data = data
        self.data_col = data_col
        self.pure_data = data[data_col]
        pass

    def data_preprocessing(self):
        print(self.pure_data)
        log_fn = FunctionTransformer(func=lambda x: np.log(x), validate=False)
        data_processed = log_fn.transform(self.pure_data)
        scaler = MinMaxScaler()
        data_processed = pd.DataFrame(scaler.fit_transform(data_processed), index=self.pure_data.index,
                                      columns=self.pure_data.columns)
        pca = PCA(n_components=4, svd_solver='full')
        data_processed = pd.DataFrame(pca.fit_transform(pd.DataFrame(data_processed)), index=self.pure_data.index,
                                      columns=self.pure_data.columns)
        return data_processed
        pass
