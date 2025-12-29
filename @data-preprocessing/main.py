import os
import sys
import argparse

# 确保可以导入 scripts/raw_to_structral_data.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # data/I-MAESTRO_data
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# 将工作目录切换到工程根目录，使相对路径 BASE_DIR='data' 等保持可用
os.chdir(PROJECT_ROOT)

from raw_to_structral_data import DataLoader, DataProcessor, DataExporter, Preprocessing, Tee

# 默认配置
DEFAULT_site = "bauges"  # 可选: bauges / milicz / sneznik
DATA_BASE_DIR = os.path.join("@data", "imaestro")  # 新 IMAESTRO 数据根目录


def setup_logging(site: str) -> None:
    """将标准输出/错误重定向到 @data-preprocessing 下的日志文件。"""
    log_dir = "@data-preprocessing"
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, f"{site}_imaestro_out.txt")
    logfile = open(out_path, "w")
    sys.stdout = Tee(sys.__stdout__, logfile)
    sys.stderr = Tee(sys.__stderr__, logfile)
    print(f"Logging to {out_path}")


def run_pipeline(site: str) -> None:
    """在 @data/imaestro/<site> 上运行结构数据处理与栅格导出。"""
    setup_logging(site)

    # 1. 加载数据（树表 + cellID25 栅格）
    loader = DataLoader(site, data_base_dir=DATA_BASE_DIR)

    # 2. 计算 biomass、物种/高度统计，并生成 <site>_cells.csv
    _ = DataProcessor(loader)

    # 3. 将关键指标导出为 GeoTIFF
    exporter = DataExporter(loader)
    exporter.write_array_to_tiff("biomass_t_ha")
    exporter.write_array_to_tiff("height95")
    exporter.write_array_to_tiff("dom_genus")

    # 4. 对导出的 GeoTIFF 进行平滑预处理
    preproc = Preprocessing(loader)
    preproc.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run I-MAESTRO structural data preprocessing on @data/imaestro."
    )
    parser.add_argument(
        "--site",
        type=str,
        default=DEFAULT_site,
        choices=["bauges", "milicz", "sneznik"],
        help="site name to process.",
    )
    args = parser.parse_args()
    run_pipeline(args.site)


if __name__ == "__main__":
    main()
