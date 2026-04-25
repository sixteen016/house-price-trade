# predict_house.py
import pandas as pd
import numpy as np
import re
import joblib
import os

# LightGBM兼容性检查
LIGHTGBM_AVAILABLE = False
MODEL_LOADED = False
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (OSError, ImportError) as e:
    if "libgomp.so.1" in str(e) or "No module named 'lightgbm'" in str(e):
        print("警告: LightGBM不可用，使用降级预测模式")
        LIGHTGBM_AVAILABLE = False
    else:
        raise

# 备用预测函数（仅作为紧急备用）
def _fallback_prediction(data):
    """当LightGBM完全不可用时的紧急备用预测"""
    # 简单的基于面积的预测（作为最后的手段）
    area = data.get('area', 80)
    base_price = area * 5000
    
    # 添加一些基本调整
    room_bonus = data.get('room', 2) * 20000
    decoration_bonus = {'简装': 0, '中装': 30000, '精装': 60000, '豪装': 100000}.get(data.get('decoration', '精装'), 0)
    
    return int(base_price + room_bonus + decoration_bonus)

# ---------- 全局缓存 ----------
_model = None
_le = None
_scaler = None
_kmeans = None
_community_mean = None
_cluster_mean = None
_global_mean = None
_feature_names = None
_base_features = None

def _load_objects():
    """加载所有保存的模型和预处理对象（仅一次）"""
    global _model, _le, _scaler, _kmeans, _community_mean, _cluster_mean, _global_mean, _feature_names, _base_features, MODEL_LOADED
    
    if MODEL_LOADED:
        return
        
    try:
        # 尝试加载预处理对象
        _le = joblib.load('lightgbm_label_encoder.pkl')
        _scaler = joblib.load('lightgbm_scaler.pkl')
        _kmeans = joblib.load('lightgbm_kmeans.pkl')
        _community_mean = joblib.load('lightgbm_community_mean.pkl')
        _cluster_mean = joblib.load('lightgbm_cluster_mean.pkl')
        _global_mean = joblib.load('lightgbm_global_mean.pkl')
        _feature_names = joblib.load('lightgbm_feature_names.pkl')
        _base_features = joblib.load('lightgbm_base_features.pkl')
        
        # 只有在LightGBM可用时才加载模型
        if LIGHTGBM_AVAILABLE:
            _model = lgb.Booster(model_file='lightgbm_house_price_model.txt')
            print("LightGBM模型加载成功")
        else:
            print("使用降级预测模式")
            
        MODEL_LOADED = True
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("将使用降级预测模式")
        MODEL_LOADED = True

# ---------- 辅助函数（与preprocess.py一致） ----------
def get_room(text):
    match = re.search(r'(\d+)\s*室', text)
    return match.group(1) if match else None

def get_hall(text):
    match = re.search(r'(\d+)\s*厅', text)
    return match.group(1) if match else None

def get_ward(text):
    match = re.search(r'(\d+)\s*卫', text)
    return match.group(1) if match else None

def get_kitchen(text):
    match = re.search(r'(\d+)\s*厨', text)
    return match.group(1) if match else None

def get_floors(text):
    match = re.search(r'共(\d+)\s*层', text)
    return match.group(1) if match else None

def chinese_to_number(s):
    if not s:
        return 0
    units = {'一':1, '二':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9}
    tens = {'十':10}
    all_map = {**units, **tens, '两':2}
    if len(s) == 1:
        return all_map.get(s, 0)
    if s[0] == '十':
        return 10 + (units.get(s[1], 0) if len(s) > 1 else 0)
    if s[-1] == '十':
        return units.get(s[0], 0) * 10
    if '十' in s:
        parts = s.split('十')
        left = units.get(parts[0], 0) * 10
        right = units.get(parts[1], 0) if len(parts) > 1 else 0
        return left + right
    return sum(all_map.get(c, 0) for c in s)

def get_elevator(text):
    pattern = r'(?:(\d+)|([一二两三四五六七八九十]+))\s*梯\s*(?:(\d+)|([一二两三四五六七八九十]+))\s*户'
    match = re.search(pattern, text)
    if match:
        ti = int(match.group(1)) if match.group(1) is not None else chinese_to_number(match.group(2))
        hu = int(match.group(3)) if match.group(3) is not None else chinese_to_number(match.group(4))
    else:
        ti, hu = None, None
    return ti, hu

# ---------- 单条数据预处理 ----------
def preprocess_single(house_dict):
    """
    输入：单个房屋的字典，包含所有原始字段（与训练数据字段一致）
    输出：预处理后的 DataFrame，列顺序与训练时完全一致
    """
    _load_objects()
    df = pd.DataFrame([house_dict])

    # 1. 数据清洗
    df = df.replace({'暂无数据': np.nan, '未知': np.nan})

    # 布尔映射
    elevator_map = {'有': 1, '无': 0}
    df['配备电梯'] = df['配备电梯'].map(elevator_map).fillna(0)
    ownership_map = {'非共有': 1, '共有': 0}
    df['房权所属'] = df['房权所属'].map(ownership_map).fillna(0)

    # 2. 房屋户型提取
    df['室'] = df['房屋户型'].apply(get_room).astype(float)
    df['厅'] = df['房屋户型'].apply(get_hall).astype(float)
    df['卫'] = df['房屋户型'].apply(get_ward).astype(float)
    df['厨'] = df['房屋户型'].apply(get_kitchen).astype(float)
    df.drop('房屋户型', axis=1, inplace=True)

    # 3. 建筑面积
    df['建筑面积'] = df['建筑面积'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

    # 4. 楼层处理
    df['总楼层'] = df['所在楼层'].apply(get_floors).astype(float)
    pattern = r'(低|中|高)楼层'
    df['楼层分类'] = df['所在楼层'].str.extract(pattern, expand=False)
    level_map = {'低': 0.2, '中': 0.5, '高': 0.8}
    df['楼层数值'] = df['楼层分类'].map(level_map).fillna(0.5)
    df['实际楼层'] = np.ceil(df['楼层数值'] * df['总楼层'])
    df.drop(['所在楼层', '楼层分类'], axis=1, inplace=True)

    # 5. 朝向处理
    df['主朝向'] = df['房屋朝向'].str.extract(r'([东西南北])', expand=False)
    df['多朝向'] = df['房屋朝向'].str.contains(r'[东西南北].*[东西南北]').astype(int)
    df.drop('房屋朝向', axis=1, inplace=True)

    # 6. 主朝向独热编码（保持与训练时相同的列）
    main_dir_cols = [col for col in _feature_names if col.startswith('主朝向_')]
    if not main_dir_cols:
        main_dir_cols = ['主朝向_东', '主朝向_南', '主朝向_西', '主朝向_北', '主朝向_nan']
    main_dir_series = df['主朝向'].fillna('nan')
    dummies = pd.get_dummies(main_dir_series, prefix='主朝向')
    for col in main_dir_cols:
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[main_dir_cols]
    df = pd.concat([df, dummies], axis=1)
    df.drop('主朝向', axis=1, inplace=True)

    # 7. 时间处理
    df['成交时间'] = pd.to_datetime(df['成交时间'], format='%Y.%m.%d', errors='coerce')
    df['挂牌时间'] = pd.to_datetime(df['挂牌时间'], format='%Y-%m-%d', errors='coerce')
    df['成交年份'] = df['成交时间'].dt.year.fillna(0).astype(int)
    df['成交月份'] = df['成交时间'].dt.month.fillna(0).astype(int)
    df['成交日'] = df['成交时间'].dt.day.fillna(0).astype(int)
    df['挂牌年份'] = df['挂牌时间'].dt.year.fillna(0).astype(int)
    df['挂牌月份'] = df['挂牌时间'].dt.month.fillna(0).astype(int)
    df['挂牌日'] = df['挂牌时间'].dt.day.fillna(0).astype(int)
    df.drop(['成交时间', '挂牌时间'], axis=1, inplace=True)

    # 8. 梯户比例
    df['梯数'], df['户数'] = zip(*df['梯户比例'].apply(get_elevator))
    df['梯数'] = df['梯数'].fillna(0)
    df['户数'] = df['户数'].fillna(0)
    df['梯户比例数值'] = df['梯数'] / df['户数'].replace(0, np.nan)
    df['梯户比例数值'].fillna(0, inplace=True)
    df.drop('梯户比例', axis=1, inplace=True)

    # 9. 经纬度
    if '百度经纬' in df.columns:
        split_df = df['百度经纬'].str.split(',', expand=True)
        df['经度'] = split_df[0].astype(float)
        df['纬度'] = split_df[1].astype(float)
        df.drop('百度经纬', axis=1, inplace=True)

    # 10. 构建最终特征DataFrame（顺序与训练时一致）
    final_df = pd.DataFrame(index=[0])

    # 数值特征（带 num__ 前缀）
    for col in _base_features:
        raw_col = col.replace('num__', '')
        if raw_col in df.columns:
            final_df[col] = df[raw_col].values
        else:
            final_df[col] = 0

    # 经纬度
    final_df['num__经度'] = df['经度'].values
    final_df['num__纬度'] = df['纬度'].values

    # 小区
    community_raw = house_dict.get('小区', '')
    try:
        community_code = _le.transform([community_raw])[0]
    except ValueError:
        community_code = -1
    final_df['小区'] = community_code
    final_df['小区_kfold_mean'] = _community_mean.get(community_raw, _global_mean)

    # 聚类
    coords = np.array([[df['经度'].values[0], df['纬度'].values[0]]])
    coords_scaled = _scaler.transform(coords)
    cluster_label = _kmeans.predict(coords_scaled)[0]
    final_df['cluster'] = cluster_label
    final_df['cluster_kfold_mean'] = _cluster_mean.get(cluster_label, _global_mean)
    distances = _kmeans.transform(coords_scaled)
    final_df['dist_to_center'] = distances.min(axis=1)[0]

    # 确保所有特征列都存在，缺失补0
    for col in _feature_names:
        if col not in final_df.columns:
            final_df[col] = 0

    # 按训练时的顺序排列
    final_df = final_df[_feature_names]
    return final_df

def predict_house_price(house_dict):
    """预测房价（万元）"""
    try:
        _load_objects()
        
        # 检查是否可以使用LightGBM模型
        if LIGHTGBM_AVAILABLE and '_model' in globals() and _model is not None:
            X = preprocess_single(house_dict)
            return np.exp(_model.predict(X)[0])
        else:
            # 如果LightGBM不可用，抛出明确错误
            raise RuntimeError("LightGBM模型不可用，无法进行准确预测")
            
    except Exception as e:
        print(f"预测过程中出错: {e}")
        # 重新抛出异常，让调用者处理
        raise

# 测试（直接运行本文件时执行）
if __name__ == '__main__':
    test_house = {
        '小区': '幕府西路85号',
        '房屋户型': '2室2厅1卫',
        '建筑面积': '85.00㎡',
        '所在楼层': '中楼层(共18层)',
        '房屋朝向': '南',
        '梯户比例': '一梯两户',
        '配备电梯': '有',
        '房权所属': '非共有',
        '成交周期（天）': 30,
        '调价（次）': 2,
        '带看（次）': 5,
        '关注（人）': 10,
        '浏览（次）': 100,
        '建成年代': 2010,
        '户型结构': '平层',
        '建筑类型': '板楼',
        '装修情况': '精装',
        '建筑结构': '钢混',
        '交易权属': '商品房',
        '房屋用途': '普通住宅',
        '成交时间': '2023.01.01',
        '挂牌时间': '2022-12-01',
        '百度经纬': '118.8,32.1'
    }
    price = predict_house_price(test_house)
    print(f"预测价格：{price:.2f} 万元")