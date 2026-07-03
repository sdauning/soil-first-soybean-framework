import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    # 保留经纬度用于后续分析，但不用于训练
    coords = data[['Longitude', 'Latitude']].copy()
    features = data.drop(['Longitude', 'Latitude', 'Yield'], axis=1)
    y = data['Yield']
    return pd.concat([coords, features, y], axis=1)


def train_and_save_model(data, model_type='rf'):
    # 分离特征和标签（不包含经纬度）
    X = data.drop(['Longitude', 'Latitude', 'Yield'], axis=1)
    y = data['Yield']

    # 处理缺失值
    X = X.fillna(X.mean())

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 保存特征名称
    feature_names = X.columns.tolist()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 训练模型
    if model_type == 'rf':
        model = RandomForestRegressor(random_state=42)
    else:
        model = LGBMRegressor(random_state=42)

    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance on Test Set:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # 保存模型和scaler
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(model, 'saved_models/yield_predictor.pkl')
    joblib.dump(scaler, 'saved_models/scaler.pkl')

    # 保存特征顺序
    with open('saved_models/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))

    return model, scaler, feature_names


def predict_new_data(new_data_path, model_path='saved_models/yield_predictor.pkl',
                     scaler_path='saved_models/scaler.pkl',
                     feature_names_path='saved_models/feature_names.txt'):
    # 加载模型和scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 加载特征顺序
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]

    # 读取新数据
    new_data = pd.read_csv(new_data_path)

    # 保留经纬度用于结果输出
    coords = new_data[['Longitude', 'Latitude']].copy()

    # 检查是否包含真实产量
    has_true_yield = 'Yield' in new_data.columns

    # 准备特征数据（按照训练时的顺序）
    try:
        X_new = new_data[feature_names].copy()
    except KeyError as e:
        missing_features = set(feature_names) - set(new_data.columns)
        raise ValueError(f"以下特征在新数据中缺失: {missing_features}") from e

    if has_true_yield:
        y_true = new_data['Yield']
    else:
        y_true = None

    # 处理缺失值
    X_new = X_new.fillna(X_new.mean())

    # 应用相同的预处理
    X_new_scaled = scaler.transform(X_new)

    # 进行预测
    y_pred = model.predict(X_new_scaled)

    # 创建包含完整信息的结果DataFrame
    results = pd.DataFrame({
        'Longitude': coords['Longitude'],
        'Latitude': coords['Latitude'],
        'Predicted_Yield': y_pred
    })

    # 如果有真实值，添加并计算性能指标
    if has_true_yield:
        results['True_Yield'] = y_true

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        print("\nPrediction Performance on New Data:")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")

    # 保存预测结果
    results.to_csv('prediction_results.csv', index=False)
    print("\nPredictions saved to 'prediction_results.csv'")

    return results


if __name__ == "__main__":
    # 训练数据路径
    train_data_path = r'C:\Users\wcx\Desktop\大豆产量与各因素的关系\finallymodel\all.csv'

    # 1. 加载并预处理数据
    train_data = load_and_preprocess_data(train_data_path)

    # 2. 训练模型并保存
    model, scaler, feature_names = train_and_save_model(train_data, model_type='rf')

    # 3. 使用新数据进行预测
    new_data_path = r'C:\Users\wcx\Desktop\大豆产量与各因素的关系\finallymodel\2008.csv'  # 替换为你的新数据文件路径
    if os.path.exists(new_data_path):
        try:
            predictions = predict_new_data(new_data_path)
            print("\nPrediction results:")
            print(predictions.head())
        except Exception as e:
            print(f"\nError during prediction: {str(e)}")
    else:
        print(f"\nNew data file not found at {new_data_path}. Please provide a valid path.")