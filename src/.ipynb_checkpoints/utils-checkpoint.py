import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


# --- (다른 유틸리티 함수들: plot_multiple_axes, resumetable, col_value_counts 등) ---
def get_data_path(file_name):
    """
    현재 스크립트 파일의 위치를 기준으로 'data/' 폴더 내의 파일 경로를 반환합니다.
    """
    current_script_dir = Path(os.path.abspath(__file__)).parent
    project_root = current_script_dir.parent # 'src'의 부모 폴더 (즉, 프로젝트 루트)
    data_folder = project_root / 'data'
    file_path = data_folder / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
    return str(file_path)

def load_train_data():
    """train.csv 파일을 로드하여 DataFrame으로 반환합니다."""
    return pd.read_csv(get_data_path('train.csv'))

def load_test_data():
    """test.csv 파일을 로드하여 DataFrame으로 반환합니다."""
    return pd.read_csv(get_data_path('test.csv'))

# 시각화 함수에 percentage 표현
def write_percent(ax, total_size):
    for patch in ax.patches:
        height = patch.get_height()
        width = patch.get_width()
        left_coord = patch.get_x()
        percent = height / total_size * 100

        ax.text(left_coord + width/2.0,
            height + total_size * 0.001,
            '{:1.1f}%'.format(percent),
            ha='center')

# Feature Table 함수
import pandas as pd
import numpy as np

def resumetable(df, missing_value=-1, ignore_cols=None, verbose=True):
    ignore_cols = ignore_cols or []
    if verbose:
        print(f'Data shape: {df.shape}')

    summary = pd.DataFrame(df.dtypes, columns=['Data Type'])
    summary['Missing'] = (df == missing_value).sum().values
    summary['Nunique'] = df.nunique().values
    summary['Feature Type'] = None

    for col in df.columns:
        if col in ignore_cols:
            continue
        if 'bin' in col:
            summary.loc[col, 'Feature Type'] = 'Binary'
        elif 'cat' in col:
            summary.loc[col, 'Feature Type'] = 'Nominal'
        elif np.issubdtype(df[col].dtype, np.floating):
            summary.loc[col, 'Feature Type'] = 'Continuous'
        elif np.issubdtype(df[col].dtype, np.integer):
            summary.loc[col, 'Feature Type'] = 'Ordinal'

    summary = summary.sort_values(by='Feature Type')
    return summary

# Value Counts 함수
# 이상치 등 다룰 때 실제 값의 개수 파악할 때 사용 -> 값 적으면 왜곡 가능하기에 반드시 확인 
def col_value_counts(df, column):
    print(f'--- {column} value_counts ---')
    print(df[column].value_counts())

# 여러개의 함수 Plot 
def plot_multiple_axes(df, cols, plot_type='hist', target=None, n_cols=3, height=4, bins=30, xrot=0, exclude_cols=None):
    """
    여러 피처의 시각화를 한 번에 출력하는 범용 함수
    특정 피처를 제외하고 시각화할 수 있는 기능을 추가.
    boxplot/violinplot 시 target을 x축으로 고정하여 분포 비교에 최적화.

    Parameters:
        df (pd.DataFrame): 데이터프레임
        cols (list): 시각화할 전체 컬럼 리스트 (exclude_cols에 따라 필터링됨)
        plot_type (str): 'hist', 'count', 'bar', 'box', 'violin' 중 하나
        target (str, optional): barplot, boxplot, violinplot일 경우 사용할 타겟 변수명.
                                 'bar'일 땐 y축, 'box'/'violin'일 땐 x축으로 사용됨.
        n_cols (int): 한 줄에 그릴 그래프 수
        height (int): 서브플롯 하나의 높이
        bins (int): histplot용 구간 수
        xrot (int): x축 레이블 회전 각도
        exclude_cols (list, optional): 시각화에서 제외할 컬럼 리스트. 기본값은 None.
    """
    # 제외할 컬럼 필터링
    # cols는 Index 타입일 수도 리스트 타입일 수도 있으므로, 리스트로 변환하여 처리.
    cols_list = cols.tolist() if isinstance(cols, pd.Index) else list(cols)

    if exclude_cols:
        cols_to_plot = [col for col in cols_list if col not in exclude_cols]
    else:
        cols_to_plot = cols_list

    # 필터링된 컬럼 리스트가 비어있는지 확인
    if not cols_to_plot: # 이 부분이 if not cols_to_plot: 이었음.
        print("시각화할 컬럼이 없습니다.")
        return

    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * height))
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        if plot_type == 'hist':
            sns.histplot(data=df, x=col, bins=bins, ax=ax) #, hue=col, palette='pastel', legend=False)
        elif plot_type == 'count':
            sns.countplot(data=df, x=col, ax=ax, hue=col, palette='pastel', legend=False)
        elif plot_type == 'bar' and target:
            sns.barplot(data=df, x=col, y=target, ax=ax, hue=col, palette='pastel', legend=False)
        elif plot_type == 'box' and target:
            sns.boxplot(data=df, x=target, y=col, ax=ax, hue=col, palette='pastel', legend=False)
        elif plot_type == 'violin' and target:
            sns.violinplot(data=df, x=target, y=col, ax=ax, hue=col, palette='pastel', legend=False)
        else:
            ax.text(0.5, 0.5, 'Invalid plot_type or missing target', ha='center')

        ax.set_title(f'{col} ({plot_type})')
        ax.tick_params(axis='x', rotation=xrot)

    # 여분 축 제거
    # 현재 i는 마지막으로 그려진 플롯의 인덱스이므로, i+1부터 제거 시작
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

