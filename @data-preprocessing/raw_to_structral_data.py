# raw_to_structral_data.py

# ====== import ======
import numpy as np
import os
import rasterio
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import contextily as ctx
import json
from pyproj import Transformer
from IPython.display import display
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter, generic_filter

import sys

# ====== 全局参数 ======
BASE_DIR = '@data'
LANDSCAPE = ('bauges', 'milicz', 'sneznik')
LANDSCAPE_CRS = {
    'bauges': {
        'crs': 'EPSG:2154',
        'XLLCORNER': 927725,
        'YLLCORNER': 6491000
    },
    'milicz': {
        'crs': 'EPSG:2180',
        'XLLCORNER': 366925,
        'YLLCORNER': 390875
    },
    'sneznik': {
        'crs': 'EPSG:3912',
        'XLLCORNER': 454000,
        'YLLCORNER': 48000
    }
}

CELL_SIZE_M = 25
CELL_AREA_HA = (CELL_SIZE_M ** 2) / 10000.0

# ====== 日志重定向 ======
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # 保证及时写入
    def flush(self):
        for f in self.files:
            f.flush()

# ====== 数据加载 ======
class DataLoader:
    def __init__(self, landscape, data_base_dir=None):
        """加载指定景观的数据。

        landscape: 景观名称，例如 'bauges' / 'milicz' / 'sneznik'
        data_base_dir: 数据根目录，默认为 BASE_DIR ('data')，
                       对 IMAESTRO 新数据可传入 '@data/imaestro' 或其绝对路径。
        """
        self.landscape = landscape
        if data_base_dir is None:
            data_base_dir = BASE_DIR
        self.data_base_dir = data_base_dir
        self.landscape_dir = os.path.join(self.data_base_dir, landscape)
        self.trees_df, self.arr = None, None
        self._load_file()
    
    def _load_file(self):
        CSV_PATH = os.path.join(self.landscape_dir, 'raw', f'{self.landscape}_trees.csv')
        tqdm.pandas(desc=f"Reading {CSV_PATH}")
        self.trees_df = pd.read_csv(CSV_PATH)
        
        ASC_PATH = os.path.join(self.landscape_dir, 'raw', f'{self.landscape}_cellID25.asc')
        with rasterio.open(ASC_PATH) as src:
            self.arr = src.read(1)

        print(f"✅{CSV_PATH} and {ASC_PATH} reading done")
        print(f"total trees: {self.trees_df.shape}")
        print(f"total cells: {self.arr.shape}")
        print("first 3 rows of trees_df:")
        print(self.trees_df.head(3))

class DataProcessor:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.wood_density = None
        self.cell_df = None
        # analysis
        self._calculate_biomass()
        self._statistics()
        self._aba_species_statistics()
        self._aba_height_statistics()
        
        # processing
        self._calculate_cell_biomass()
        self._calculate_cell_height()
        self._save_cells()

    def _calculate_biomass(self, allometric_file=os.path.join(BASE_DIR, 'auxiliary/wood_density_allometric.json'), default_density=0.55, default_a=0.0673, default_b=2.5):
        """
        for each tree, use allometric equation: biomass (kg) = a * (dbh_cm ** b) * wood_density
        dbh_cm = self.trees_df['dbh'] (cm)
        species name: self.trees_df['sp']
        """
        with open(allometric_file, 'r') as f:
            allometric_dict = json.load(f)
        
        def compute_row_biomass(row):
            sp = row['sp']
            dbh_cm = row['dbh']
            params = allometric_dict.get(sp, None)
            if params is not None:
                wd, a, b = params['wood_density'], params['a'], params['b']
            else:
                wd, a, b = default_density, default_a, default_b
            if pd.isnull(dbh_cm):
                return np.nan
            return a * (dbh_cm ** b) * wd
        
        self.loader.trees_df['biomass'] = self.loader.trees_df.apply(compute_row_biomass, axis=1)
        print("✅Biomass calculated and added to table.")
        print(self.loader.trees_df.head(3))

    def _statistics(self):
        print("✅Statistics for trees data:")
        print(self.loader.trees_df.describe())

    def _aba_species_statistics(self):
        """
        按cellID25分组，统计：
        - 主导树种（数量最多的sp）
        - Shannon多样性指数
        - 各树种比例
        - 主导属（genus）
        返回：cellID25为index的DataFrame
        """
        self.loader.trees_df['genus'] = self.loader.trees_df['sp'].apply(lambda x: str(x).split()[0] if pd.notnull(x) else np.nan)
        cell_stats = []
        for cell, group in self.loader.trees_df.groupby('cellID25'):
            sp_counts = group['sp'].value_counts() # 按cellID25分组，统计每个cellID25中每个树种的数量
            genus_counts = group['genus'].value_counts() # 按cellID25分组，统计每个cellID25中每个属的数量
            total = sp_counts.sum() # 按cellID25分组，统计每个cellID25中所有树种的数量
            sp_props = (sp_counts / total).round(3).to_dict()  # 按cellID25分组，统计每个cellID25中每个树种的比例，保留3位小数
            genus_props = (genus_counts / total).round(3).to_dict() # 按cellID25分组，统计每个cellID25中每个属的比例，保留3位小数
            dominant_sp = sp_counts.idxmax() if not sp_counts.empty else None # 按cellID25分组，统计每个cellID25中数量最多的树种
            dominant_genus = genus_counts.idxmax() if not genus_counts.empty else None # 按cellID25分组，统计每个cellID25中数量最多的属
            cell_stats.append({
                'cellID25': cell,
                'dom_spe': dominant_sp,
                'dom_genus': dominant_genus,
                'sp_props': sp_props,
                'genus_props': genus_props,
                'n_trees': total
            })
        cell_stats_df = pd.DataFrame(cell_stats).set_index('cellID25')
        out_path = os.path.join(self.loader.landscape_dir, 'aba_species_statistics.csv')
        cell_stats_df.to_csv(out_path)
        print(f"\n✅ABA species statistics saved to {out_path}")
        print(cell_stats_df.drop(columns=['sp_props', 'genus_props']).head().T)
        
    def _aba_height_statistics(self):
        """
        对每个cell，基于h和n列，计算高度的最小值、95百分位、99百分位、最大值和平均值，
        同时记录该cell中数量最多（按n求和）的树种及其数量。
        """
        results = []
        for cell, group in self.loader.trees_df.groupby('cellID25'):
            expanded = group.loc[group.index.repeat(group['n'])].copy()
            expanded = expanded.sort_values('h').reset_index(drop=True)
            n_total = len(expanded)
            if n_total == 0:
                continue

            heights = expanded['h'].values
            # 基本统计量
            height_min = float(np.min(heights))
            height_max = float(np.max(heights))
            height_mean = float(np.mean(heights))

            # 百分位高度（按照离散排序索引计算）
            def _percentile(sorted_values, q):
                idx = int(np.ceil(q * len(sorted_values))) - 1
                idx = max(0, min(idx, len(sorted_values) - 1))
                return float(sorted_values[idx])

            height_p95 = _percentile(heights, 0.95)
            height_p99 = _percentile(heights, 0.99)

            results.append({
                'cellID25': cell,
                'height_min': height_min,
                'height_mean': height_mean,
                'height_max': height_max,
                'height95': height_p95,
                'height99': height_p99
            })
        df = pd.DataFrame(results)
        out_path = os.path.join(self.loader.landscape_dir, 'aba_height_statistics.csv')
        df.to_csv(out_path)
        print(f"\n✅ABA height percentile/min/max/mean statistics saved to {out_path}")
        print(df.head().T)

    def _calculate_cell_biomass(self):
        """
        对每个cell，按n加权求和biomass，并换算为t/ha，
        生成cell级别表，包含cellID25和biomass_t_ha。
        结果存为self._milicz_cells。
        """
        df = self.loader.trees_df.copy()
        if 'biomass' not in df.columns:
            self._calculate_biomass()
        # 计算每行的总biomass（考虑n）
        df['biomass_total'] = df['biomass'] * df['n']
        cell_biomass = df.groupby('cellID25')['biomass_total'].sum().reset_index()
        # biomass_total 单位为 kg，将其换算为 t/ha
        cell_biomass['biomass_t_ha'] = (cell_biomass['biomass_total'] / 1000.0) / CELL_AREA_HA
        cell_biomass['biomass_t_ha'] = cell_biomass['biomass_t_ha'].round(3)
        cell_biomass = cell_biomass[['cellID25', 'biomass_t_ha']]
        self.cell_df = cell_biomass
    
    def _calculate_cell_height(self, height_csv=None):
        """
        读取height csv和物种统计csv，将95分位高度和主导属(dom_genus)加入cells表。
        """
        if height_csv is None:
            height_csv = os.path.join(self.loader.landscape_dir, 'aba_height_statistics.csv')
        df_height = pd.read_csv(height_csv)
        # 使用95分位树高，并与物种统计表中的dom_genus合并
        df_height = df_height[['cellID25', 'height95']]

        species_csv = os.path.join(self.loader.landscape_dir, 'aba_species_statistics.csv')
        df_species = pd.read_csv(species_csv)
        df_species = df_species[['cellID25', 'dom_genus']]

        self.cell_df = pd.merge(self.cell_df, df_height, on='cellID25', how='left')
        self.cell_df = pd.merge(self.cell_df, df_species, on='cellID25', how='left')
    
    def _save_cells(self):
        out_path = os.path.join(self.loader.landscape_dir, f'{self.loader.landscape}_cells.csv')
        # 最终仅保留指定列
        cols = ['cellID25', 'biomass_t_ha', 'height95', 'dom_genus']
        self.cell_df = self.cell_df[cols]
        self.cell_df.to_csv(out_path)
        print(f"\n✅cell_df saved to {out_path}")
        print(self.cell_df.head()) 

    @staticmethod
    def milicz_cells_analysis(landscape):
        """
        读取milicz_cells.csv，分析biomass_sum与rh95_n的关系，并统计genus个数、每个genus的count和占比。
        """
        df = pd.read_csv(os.path.join(BASE_DIR, landscape, f'{landscape}_cells.csv'))
        print(f"\n✅Analyzing {landscape} {landscape}_cells.csv")
        # 分析genus个数
        genus_counts = df['height_genus'].value_counts()
        genus_num = genus_counts.size
        genus_ratio = (genus_counts / genus_counts.sum()).round(3)
        print(f"Total {genus_num} genera")
        print("Each genus count:")
        print(genus_counts)
        print("Each genus ratio:")
        print(genus_ratio)


class DataExporter:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        # 输出路径与数据根目录保持一致（原始数据在 data 下，新数据在 @data/imaestro 下）
        self.output_dir = os.path.join(self.loader.data_base_dir, self.loader.landscape, 'output_tiff')
        os.makedirs(self.output_dir, exist_ok=True)
        self.cell_df = pd.read_csv(os.path.join(self.loader.data_base_dir, self.loader.landscape, f'{self.loader.landscape}_cells.csv'))
    
    def _fill_arr(self, column_name):
        if column_name in ['height_genus', 'dom_genus']:
            genus_map = json.load(open(os.path.join(BASE_DIR, 'auxiliary/genus_map.json')))
            mapped_col = f"{column_name}_mapped"
            self.cell_df[mapped_col] = self.cell_df[column_name].map(genus_map)
            column_name = mapped_col
        column_dict = self.cell_df.set_index('cellID25')[column_name].to_dict()
        filled_array = np.vectorize(column_dict.get)(self.loader.arr)
        if filled_array.dtype == object:
            filled_array = np.where(filled_array == None, np.nan, filled_array).astype(float)
        print(f"\n✅ {column_name} filled into .asc map")

        plt.figure(figsize=(8, 6))
        plt.imshow(filled_array, cmap='viridis')
        plt.colorbar(label=column_name)
        plt.title(column_name)
        plt.xlabel('width')
        plt.ylabel('height')
        plt.show()

        return filled_array
    
    def write_array_to_tiff(self, column_name):
        """
        将cell_df[column_name]按cellID25映射到arr，并写入tiff
        """
        XLLCORNER = LANDSCAPE_CRS[self.loader.landscape]['XLLCORNER']
        YLLCORNER = LANDSCAPE_CRS[self.loader.landscape]['YLLCORNER']
        CRS = LANDSCAPE_CRS[self.loader.landscape]['crs']
        array = self._fill_arr(column_name)
        transform = rasterio.transform.from_origin(XLLCORNER, YLLCORNER + array.shape[0] * 25, 25, 25)
        output_path = self.output_dir + f'/{self.loader.landscape}_{column_name}.tif'
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=array.dtype,
            crs=CRS,
            transform=transform,
        ) as dst:
            dst.write(array, 1)
        print(f"\n✅ Data written to TIFF successfully at {output_path}.")


class Preprocessing:
    """对导出的 GeoTIFF 进行平滑处理。

    - 连续变量（biomass_t_ha, height95）使用高斯平滑，并保证平滑后值不会比原值小太多；
    - 分类变量（dom_genus，对应数值编码）使用多数表决平滑边界。
    """

    def __init__(self, loader: DataLoader, window_size: int = 5):
        self.loader = loader
        self.window_size = window_size
        self.output_dir = os.path.join(self.loader.data_base_dir, self.loader.landscape, 'output_tiff')

    def _get_tiff_path(self, column_name: str) -> str:
        return os.path.join(self.output_dir, f"{self.loader.landscape}_{column_name}.tif")

    def _get_smooth_path(self, column_name: str) -> str:
        return os.path.join(self.output_dir, f"{self.loader.landscape}_{column_name}_smooth.tif")

    def _smooth_continuous(self, column_name: str, window_size: int = None) -> None:
        """对连续字段做**局部异常点滤波**：

        - 仅在原始有效像元上调整数值，NoData 保持 NoData（不扩张地图）。
        - 只修正相对于邻域明显异常的像元，保留大尺度梯度和整体结构，避免整幅图被抹平。
        """
        in_path = self._get_tiff_path(column_name)
        if not os.path.exists(in_path):
            print(f" Continuous TIFF not found, skip: {in_path}")
            return

        with rasterio.open(in_path) as src:
            arr = src.read(1).astype(float)
            profile = src.profile

        valid = ~np.isnan(arr)
        if not valid.any():
            print(f" No valid data in {in_path}, skip smoothing.")
            return

        if window_size is None:
            window_size = self.window_size

        def _edge_preserving(values: np.ndarray) -> float:
            """局部边缘保留滤波：

            - 如果中心像元为 NoData，则保持 NoData；
            - 使用邻域的中位数和 MAD 判断中心是否为异常值；
            - 只有明显偏离邻域分布时，才将其向邻域中位数拉近一部分。
            """
            center_idx = len(values) // 2
            center = values[center_idx]
            if np.isnan(center):
                return np.nan

            neigh = values[~np.isnan(values)]
            if neigh.size < 3:
                return center

            median = np.median(neigh)
            mad = np.median(np.abs(neigh - median))
            if mad == 0:
                # 邻域非常一致，认为没有噪声，保留中心值
                return center

            sigma = 1.4826 * mad  # 将 MAD 近似转换为标准差
            diff = center - median

            # 只有明显偏离邻域（> 2 * sigma）时才视为噪声
            if np.abs(diff) <= 2.0 * sigma:
                return center

            # 异常值：向邻域中位数拉近一半，仍保留部分原始信息
            alpha = 0.5
            new_val = center - alpha * diff

            # 限制在邻域最小/最大值范围内，避免产生新的极端值
            lo, hi = neigh.min(), neigh.max()
            new_val = max(min(new_val, hi), lo)
            return float(new_val)

        smoothed = generic_filter(
            arr,
            _edge_preserving,
            size=window_size,
            mode='nearest'
        )

        # 只在原始有效像元上保留结果，其余仍为 NoData，避免地图扩张
        smoothed[~valid] = np.nan

        out_profile = profile.copy()
        out_profile.update(count=1)

        out_path = self._get_smooth_path(column_name)
        with rasterio.open(out_path, 'w', **out_profile) as dst:
            dst.write(smoothed.astype(profile['dtype']), 1)
        print(f"\n✅ Smoothed continuous TIFF written: {out_path}")

    def _majority_filter(self, data: np.ndarray) -> np.ndarray:
        """对分类数据应用多数表决滤波（忽略 NaN）。"""

        def _mode_func(values):
            vals = values[~np.isnan(values)]
            if vals.size == 0:
                return np.nan
            vals_int = vals.astype(int)
            counts = np.bincount(vals_int)
            return float(np.argmax(counts))

        size = self.window_size
        return generic_filter(data, _mode_func, size=size, mode='nearest')

    def _smooth_categorical(self, column_name: str) -> None:
        """对分类字段（数值编码）应用多数表决平滑。"""
        in_path = self._get_tiff_path(column_name)
        if not os.path.exists(in_path):
            print(f" Categorical TIFF not found, skip: {in_path}")
            return

        with rasterio.open(in_path) as src:
            arr = src.read(1).astype(float)
            profile = src.profile

        valid = ~np.isnan(arr)
        if not valid.any():
            print(f" No valid data in {in_path}, skip smoothing.")
            return

        smoothed = self._majority_filter(arr)

        # 只在原始有效像元上保留结果，其余保持 NoData，避免地图扩张
        smoothed[~valid] = np.nan

        out_profile = profile.copy()
        out_profile.update(count=1)

        out_path = self._get_smooth_path(column_name)
        with rasterio.open(out_path, 'w', **out_profile) as dst:
            dst.write(smoothed.astype(profile['dtype']), 1)
        print(f"\n✅ Smoothed categorical TIFF written: {out_path}")

    def run(self) -> None:
        """对三个关键指标的 TIFF 进行平滑后处理。"""
        # biomass_t_ha：连续变量，仅修正局部异常点
        self._smooth_continuous('biomass_t_ha')
        # height95：连续变量，仅修正局部异常点
        self._smooth_continuous('height95')
        # dom_genus：分类变量，使用多数表决
        self._smooth_categorical('dom_genus')
