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
        
        urea_crops = ['Millets', 'Barley', 'Paddy', 'Cotton']
        used_conditions = (X_new['Crop Type'].isin(urea_crops)) & X_new['Soil Type'] == 'Sandy'
        X_new['is_used_conditions_for_Urea'] = used_conditions.astype(int)
        
        DAP_conditions = (
            (X_new['Crop Type'] == 'Paddy') & (X_new['Soil Type'] == 'Sandy')
        ) | (
            (X_new['Crop Type'] == 'Tobacco') & (X_new['Soil Type'] == 'Black')
        ) | (
            (X_new['Crop Type'] == 'Maize') & (X_new['Soil Type'] == 'Sandy')
        ) | (
            (X_new['Crop Type'] == 'Wheat') & (X_new['Soil Type'] == 'Red')
        )

        X_new['is_used_conditions_for_DAP'] = DAP_conditions.astype(int)

        
#         exception_conditions = (X_new['Crop Type'].isin(['Pulses', 'Wheat'])) | (X_new['Soil Type'] == 'Black')
#         X_new['is_exception_for_28-28'] = exception_conditions.astype(int)
                
        return X_new

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        self.encoding_maps = {}

    def fit(self, X, y):
        data = pd.concat([X, y], axis=1)
        target_col = y.name # 타겟 컬럼 이름 가져오기

        # 지정된 모든 범주형 피처에 대해 루프를 돌며 인코딩 맵 생성
        for col in self.categorical_features:
            self.encoding_maps[col] = data.groupby(col)[target_col].value_counts(normalize=True).unstack().fillna(0)
        
        return self

    def transform(self, X):
        X_new = X.copy()
        for col in self.categorical_features:
            # fit에서 만든 해당 피처의 인코딩 맵을 가져옴
            encoding_map = self.encoding_maps[col]
            
            # 매핑을 통해 새로운 피처들을 생성
            encoded_features = X_new[col].map(encoding_map.to_dict('index'))
            
            # 새로운 피처들의 이름 지정 (예: crop_type_target_enc_Urea)
            encoded_df = pd.DataFrame(encoded_features.tolist(), index=X_new.index)
            encoded_df.columns = [f'{col}_target_enc_{c}' for c in encoding_map.columns]
            
            # 기존 데이터프레임과 합치기
            X_new = pd.concat([X_new, encoded_df], axis=1)
        
        # 원본 범주형 컬럼은 이제 필요 없으니 삭제
        X_new = X_new.drop(columns=self.categorical_features)
        
        return X_new

'''
class RatioFeatureGenerator(BaseEstimator, TransformerMixin):
    def init(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        NPP_sum = X_new['Nitrogen'] + X_new['Potassium'] + X_new['Phosphorous']
        
        X_new['Ni_ratio'] = X_new['Nitrogen'] / NPP_sum
        X_new['Po_ratio'] = X_new['Potassium'] / NPP_sum
        X_new['Ph_ratio'] = X_new['Phosphorous'] / NPP_sum
        return X_new 
'''

class GroupStatsFeatureGenerator(BaseEstimator, TransformerMixin):
    # 이제 여러 개의 그룹과 여러 개의 수치형 컬럼을 처리하도록 리스트를 받는다
    def __init__(self, group_by_cols, numerical_cols):
        self.group_by_cols = group_by_cols
        self.numerical_cols = numerical_cols
        self.stats_df = None

    def fit(self, X, y=None):
        # 훈련 데이터만으로 통계량을 계산하고, 나중에 변환을 위해 저장
        data = X.copy()
        
        # self.numerical_cols는 이제 리스트이므로 그대로 사용 가능
        self.stats_df = data.groupby(self.group_by_cols)[self.numerical_cols].agg(['mean', 'std']).reset_index()
        
        # 컬럼 이름 재설정 로직도 거의 그대로 사용 가능
        group_by_str = '_'.join(self.group_by_cols)
        # 그룹 컬럼 이름을 합쳐서 사용
        
        new_cols = []
        for col in self.stats_df.columns:
            if col[1]: # 멀티인덱스의 두 번째 레벨 이름이 존재하면 (mean, std) 
                new_cols.append(f'{col[0]}_{col[1]}_by_{group_by_str}')
            else: # 멀티인덱스의 두 번째 레벨 이름이 없으면 (group_by_cols)
                new_cols.append(col[0])
        self.stats_df.columns = new_cols
        
        return self

    def transform(self, X):
        X_new = X.copy()
        
        # fit에서 계산해둔 통계량을 원본 데이터에 병합(merge)
        # merge할 때도 on=self.group_by_cols 사용
        X_new = pd.merge(X_new, self.stats_df, on=self.group_by_cols, how='left')
        
        return X_new
