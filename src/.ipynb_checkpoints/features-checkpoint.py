from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class InteractionFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # fit 단계에서는 아무것도 학습할 필요가 없으므로 self를 그대로 반환
        return self

    def transform(self, X):
        # 입력받은 데이터프레임 X를 복사해서 원본을 보호
        X_new = X.copy()
        
        # 피처 엔지니어링
        special_crops = ['Paddy', 'Pulses', 'Cotton']
        X_new['is_special_crops'] = X_new['Crop Type'].isin(special_crops).astype(int)
        X_new['Moisture_x_SpecialCrops'] = X_new['Moisture'] * X_new['is_special_crops']

        return X_new
