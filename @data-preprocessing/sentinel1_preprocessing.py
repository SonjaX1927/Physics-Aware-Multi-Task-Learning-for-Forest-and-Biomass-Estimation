import os
import sys
import argparse
import subprocess

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import rasterio.transform
from pyproj import Transformer

# 复用结构变量脚本中的坐标系和栅格信息
from raw_to_structral_data import LANDSCAPE_CRS, CELL_SIZE_M, Tee

# 工程和数据根目录（绝对路径）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # /Volumes/.../data/I-MAESTRO_data
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# 与结构数据脚本保持一致：工作目录切到工程根目录
os.chdir(PROJECT_ROOT)

# 使用绝对路径指向 IMAESTRO 数据根目录，避免 SNAP gpt 在相对路径解析时出错
DATA_BASE_DIR = os.path.join(PROJECT_ROOT, "@data", "imaestro")
S1_GRAPH_PATH = os.path.join(CURRENT_DIR, "S1_GRD_SIGMA0_TC.xml")
NODATA_VALUE = -9999.0


def setup_logging(site: str) -> None:
    """将日志写入 @data-preprocessing 下的 sentinel1 日志文件。"""
    log_dir = "@data-preprocessing"
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, f"{site}_sentinel1_out.txt")
    logfile = open(out_path, "w")
    sys.stdout = Tee(sys.__stdout__, logfile)
    sys.stderr = Tee(sys.__stderr__, logfile)
    print(f"Logging to {out_path}")


def _find_safe_dir(site: str) -> str:
    """在 @data/imaestro/{site}/raw 下查找第一个 .SAFE 目录。"""
    raw_dir = os.path.join(DATA_BASE_DIR, site, "raw")
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"RAW directory not found: {raw_dir}")

    candidates = [d for d in os.listdir(raw_dir) if d.endswith(".SAFE")]
    if not candidates:
        raise FileNotFoundError(f"No .SAFE folder found under {raw_dir}")

    # 简单起见：默认使用找到的第一个 SAFE 产品
    safe_dir = os.path.join(raw_dir, candidates[0])
    return safe_dir


def run_s1_preprocessing_with_gpt(site: str) -> str:
    """调用 SNAP gpt，运行 S1_GRD_SIGMA0_TC.xml，将 SAFE 处理为地形校正后的 sigma0 GeoTIFF。

    返回：Terrain-Correction 后的临时 GeoTIFF 路径（含 VV/VH 两个波段）。
    """
    # 对 SNAP 而言，推荐直接传入 .SAFE 目录，让 Sentinel-1 读取器自行解析 manifest 和内部结构
    safe_dir = _find_safe_dir(site)
    if not os.path.exists(safe_dir):
        raise FileNotFoundError(f"SAFE product folder not found: {safe_dir}")

    tmp_dir = os.path.join(DATA_BASE_DIR, site, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    out_tc = os.path.join(tmp_dir, f"{site}_sigma0_tc.tif")

    if not os.path.exists(S1_GRAPH_PATH):
        raise FileNotFoundError(f"SNAP graph XML not found: {S1_GRAPH_PATH}")

    # 计算当前 site 在 WGS84 下的研究区域多边形，用于 Terrain-Correction 裁剪
    geo_region_wkt = _get_site_geo_region_wkt(site)

    gpt_cmd = os.environ.get("SNAP_GPT", "gpt")
    cmd = [
        gpt_cmd,
        S1_GRAPH_PATH,
        f"-Pinput={safe_dir}",
        f"-Poutput={out_tc}",
        f"-PgeoRegion={geo_region_wkt}",
    ]

    print("Running SNAP gpt command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\n✅ SNAP preprocessing finished: {out_tc}")
    return out_tc


def _get_target_grid(site: str):
    """根据 {site}_cellID25.asc 和 LANDSCAPE_CRS 构建 25m 目标栅格 (transform, crs, shape)。"""
    asc_path = os.path.join(DATA_BASE_DIR, site, "raw", f"{site}_cellID25.asc")
    if not os.path.exists(asc_path):
        raise FileNotFoundError(f"cellID25 asc not found: {asc_path}")

    with rasterio.open(asc_path) as src:
        height, width = src.height, src.width

    xll = LANDSCAPE_CRS[site]["XLLCORNER"]
    yll = LANDSCAPE_CRS[site]["YLLCORNER"]
    # 注意：from_origin 使用的是左上角坐标
    transform = rasterio.transform.from_origin(
        xll,
        yll + height * CELL_SIZE_M,
        CELL_SIZE_M,
        CELL_SIZE_M,
    )
    dst_crs = LANDSCAPE_CRS[site]["crs"]
    return transform, dst_crs, height, width


def _get_site_geo_region_wkt(site: str) -> str:
    """根据 25m grid 的范围，在本地坐标系下计算研究区 bbox，并转换到 WGS84，返回 WKT POLYGON。

    这样可以在 SNAP 的 Terrain-Correction 中通过 geoRegion 参数仅处理研究区域，避免输出整幅轨道。
    """
    asc_path = os.path.join(DATA_BASE_DIR, site, "raw", f"{site}_cellID25.asc")
    if not os.path.exists(asc_path):
        raise FileNotFoundError(f"cellID25 asc not found for geoRegion: {asc_path}")

    with rasterio.open(asc_path) as src:
        height, width = src.height, src.width

    xll = LANDSCAPE_CRS[site]["XLLCORNER"]
    yll = LANDSCAPE_CRS[site]["YLLCORNER"]
    cell = CELL_SIZE_M

    xmin = xll
    ymin = yll
    xmax = xll + width * cell
    ymax = yll + height * cell

    src_crs = LANDSCAPE_CRS[site]["crs"]
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    # 以 xmin/ymin 为起点，顺时针构造多边形
    lon1, lat1 = transformer.transform(xmin, ymin)
    lon2, lat2 = transformer.transform(xmax, ymin)
    lon3, lat3 = transformer.transform(xmax, ymax)
    lon4, lat4 = transformer.transform(xmin, ymax)

    wkt = (
        f"POLYGON(({lon1} {lat1}, {lon2} {lat2}, {lon3} {lat3}, "
        f"{lon4} {lat4}, {lon1} {lat1}))"
    )
    return wkt


def _extract_and_reproject(site: str, src_tif: str) -> None:
    """从 SNAP 结果中提取 VV/VH 波段，并重投影到 site 的 25m grid，写出 GeoTIFF。"""
    dst_transform, dst_crs, height, width = _get_target_grid(site)

    with rasterio.open(src_tif) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_nodata = src.nodata
        descriptions = src.descriptions or ()

        band_map = {}
        for idx, desc in enumerate(descriptions, start=1):
            if not desc:
                continue
            d = desc.upper()
            if "VV" in d and "VH" not in d:
                band_map.setdefault("vv", idx)
            elif "VH" in d:
                band_map.setdefault("vh", idx)

        # 如果通过名称没有识别到，则按照常见顺序退化为 1:VV, 2:VH
        if "vv" not in band_map and src.count >= 1:
            band_map["vv"] = 1
        if "vh" not in band_map and src.count >= 2:
            band_map.setdefault("vh", 2)

        arrays = {}
        for pol, bidx in band_map.items():
            data = src.read(bidx).astype("float32")
            dst = np.full((height, width), NODATA_VALUE, dtype="float32")
            reproject(
                source=data,
                destination=dst,
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=src_nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=NODATA_VALUE,
                resampling=Resampling.bilinear,
            )
            arrays[pol] = dst

    out_dir = os.path.join(DATA_BASE_DIR, site, "output_backscatter")
    os.makedirs(out_dir, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": NODATA_VALUE,
    }

    if "vv" in arrays:
        out_vv = os.path.join(out_dir, f"{site}_vv.tif")
        lin = arrays["vv"]
        db = np.full(lin.shape, NODATA_VALUE, dtype="float32")
        valid = (lin > 0) & (lin != NODATA_VALUE)
        db[valid] = 10.0 * np.log10(lin[valid])
        with rasterio.open(out_vv, "w", **profile) as dst:
            dst.write(db, 1)
        print(f"\n✅ VV backscatter (dB) written: {out_vv}")

    if "vh" in arrays:
        out_vh = os.path.join(out_dir, f"{site}_vh.tif")
        lin = arrays["vh"]
        db = np.full(lin.shape, NODATA_VALUE, dtype="float32")
        valid = (lin > 0) & (lin != NODATA_VALUE)
        db[valid] = 10.0 * np.log10(lin[valid])
        with rasterio.open(out_vh, "w", **profile) as dst:
            dst.write(db, 1)
        print(f"\n✅ VH backscatter (dB) written: {out_vh}")


def run_pipeline(site: str) -> None:
    setup_logging(site)

    # 1. SAFE -> sigma0 (SNAP gpt)
    tc_tif = run_s1_preprocessing_with_gpt(site)

    # 2. 重投影+裁剪到 25m grid，并按要求输出 VV/VH
    _extract_and_reproject(site, tc_tif)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Sentinel-1 GRD preprocessing (sigma0 + 25m grid reprojection) for I-MAESTRO sites.",
    )
    parser.add_argument(
        "--site",
        type=str,
        default="bauges",
        choices=["bauges", "milicz", "sneznik"],
        help="site name to process.",
    )
    args = parser.parse_args()
    run_pipeline(args.site)


if __name__ == "__main__":
    main()
