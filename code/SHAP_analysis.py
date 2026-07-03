# 导入所有必需的库
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
import matplotlib
import os
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use('TkAgg')

# --- 用于寻找"拐点"的辅助函数 ---
from scipy.signal import savgol_filter


def find_knee_point(x_data, y_data, window_length=5, polyorder=2):
    """
    通过基于曲率的方法寻找曲线上趋势变化最显著的点（即"拐点"或"膝点"）。
    """
    if len(x_data) < window_length:
        print(f"警告：数据点数 ({len(x_data)}) 小于 Savitzky-Golay 滤波器窗口长度 ({window_length})。正在返回中位数。")
        return np.median(x_data)

    if window_length % 2 == 0:
        window_length += 1

    if polyorder >= window_length:
        polyorder = window_length - 1
        if polyorder < 1:
            polyorder = 1

    y_second_deriv = savgol_filter(y_data, window_length, polyorder, deriv=2)
    knee_index = np.argmax(np.abs(y_second_deriv))
    sorted_x = np.array(x_data)[np.argsort(x_data)]
    return sorted_x[knee_index]


# --------------------------------------------------------------------------
# 步骤 1: 加载本地 Excel 数据
# --------------------------------------------------------------------------
print("--> 正在加载本地 Excel 数据...")
excel_file_path = r'C:\Users\lenovo\Desktop\大豆产量与各因素的关系\finallymodel\all.xlsx'
if not os.path.exists(excel_file_path):
    raise FileNotFoundError(f"未找到文件：{excel_file_path}。请检查文件路径是否正确。")
df = pd.read_excel(excel_file_path)
print(f"--> 已成功加载文件: {excel_file_path}")
print("数据预览:")
print(df.head())

# 定义特征列和目标列
target_column = 'Yield'
feature_columns = ['CO2', 'RAD', 'PPT', 'TEMP', 'BDOD', 'CEC', 'CFVO', 'CLY', 'N', 'pH', 'SND', 'SLT', 'SOC']
X = df[feature_columns]
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## 步骤 2: 训练随机森林模型
print("--> 正在训练随机森林模型...")
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print("\n--> 模型训练完成！")

# --------------------------------------------------------------------------
# 步骤 3: 计算 SHAP 值
# --------------------------------------------------------------------------
print("--> 正在计算 SHAP 值...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# --------------------------------------------------------------------------
# 步骤 4: 分别保存摘要图和依赖图
# --------------------------------------------------------------------------
print("--> 正在分别保存 SHAP 摘要图和依赖图...")

# ==================== 美学参数配置区 ====================
aesthetic_params = {
    'suptitle_size': 22,
    'ax_label_size': 16,
    'tick_label_size': 16,
    'legend_size': 16,
    'cbar_label_size': 16,
    'summary_cbar_width': 0.015,
    'summary_cbar_height_shrink': 1.0,
    'summary_cbar_pad': 0.01,
    'dep_cbar_width': 0.005,
    'dep_cbar_height_shrink': 1.0,
    'dep_cbar_pad': 0.002,
    'dep_cbar_tick_length': 1,
    'grid_wspace': 0.45,
    'grid_hspace': 0.4
}

# ==================== 颜色配置区 ====================
# 创建自定义颜色映射 - 用于依赖图的渐变色
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_blue_red",
    ['#c9453a', '#ef988b', '#90ee90', '#76c776', '#4caf50', '#2e8b57', '#39972d', '#226e22'])

color_params = {
    # 摘要图保持原样
    'summary_bar_color': "lightgray",
    'summary_bar_alpha': 0.6,
    'summary_scatter_cmap': "plasma",
    'summary_scatter_alpha': 0.8,
    'summary_scatter_size': 15,
    'grid_color': 'gray',
    'grid_alpha': 0.6,

    # 依赖图使用自定义渐变色
    'dep_scatter_cmap': custom_cmap,  # 使用自定义渐变色
    'dep_scatter_alpha': 0.8,
    'dep_scatter_size': 25,
    'median_line_color': 'black',
    'threshold_line_color': 'red',
}
# ====================================================

# --- 设置全局字体为 "Arial" ---
plt.rcParams['font.family'] = 'Arial'

# --- 计算归一化的特征重要性 ---
mean_abs_shaps = np.abs(shap_values.values).mean(axis=0)
normalized_mean_abs_shaps = mean_abs_shaps / np.sum(mean_abs_shaps)  # 归一化处理

# 创建特征重要性 DataFrame
feature_importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': normalized_mean_abs_shaps
}).sort_values('importance', ascending=True)

# --- 保存摘要图 (保持原样不变) ---
fig_summary, ax_main = plt.subplots(figsize=(8, 16))

ax_main.set_yticks(range(len(feature_importance_df)))
ax_main.set_yticklabels(feature_importance_df['feature'], fontsize=aesthetic_params['tick_label_size'])

ax_top = ax_main.twiny()
ax_top.barh(
    range(len(feature_importance_df)),
    feature_importance_df['importance'],
    color=color_params['summary_bar_color'],
    alpha=color_params['summary_bar_alpha'],
    height=0.7
)

ax_top.set_xlabel("Relative importance of environmental variables influencing soybean yield",
                  fontsize=aesthetic_params['ax_label_size'])
ax_top.tick_params(axis='x', labelsize=aesthetic_params['tick_label_size'])
ax_top.grid(False)

# 摘要图使用原来的颜色映射
cmap = plt.get_cmap(color_params['summary_scatter_cmap'])
scatter_plots = []
for i, feature_name in enumerate(feature_importance_df['feature']):
    original_idx = X_test.columns.get_loc(feature_name)
    shap_vals_for_feature = shap_values.values[:, original_idx]
    feature_vals_for_color = X_test.iloc[:, original_idx]
    y_jitter = np.random.normal(0, 0.08, shap_vals_for_feature.shape[0])
    scatter = ax_main.scatter(
        shap_vals_for_feature, i + y_jitter, c=feature_vals_for_color,
        cmap=cmap, s=color_params['summary_scatter_size'],
        alpha=color_params['summary_scatter_alpha'], zorder=1
    )
    scatter_plots.append(scatter)

# 添加颜色条
cbar = plt.colorbar(scatter_plots[0], ax=ax_main, orientation="vertical", pad=0.02)
cbar.set_label('Feature Value', fontsize=aesthetic_params['ax_label_size'])
cbar.ax.tick_params(labelsize=aesthetic_params['tick_label_size'])

ax_main.set_xlabel("SHAP value (impact on model output)", fontsize=aesthetic_params['ax_label_size'])
ax_main.tick_params(axis='x', labelsize=aesthetic_params['tick_label_size'])
ax_main.grid(True, axis='x', linestyle='--',
             color=color_params['grid_color'],
             alpha=color_params['grid_alpha'])

# 保存摘要图
output_summary_path = r'C:\Users\lenovo\Desktop\shap_summary_plot.png'
plt.savefig(output_summary_path, dpi=300, bbox_inches='tight')
print(f"--> SHAP 摘要图已成功保存到文件: {output_summary_path}")
plt.close(fig_summary)

# --- 保存依赖图 (使用自定义渐变色) ---
top_6_features = feature_importance_df['feature'].tail(6).iloc[::-1].tolist()
for i, feature in enumerate(top_6_features):
    fig_dep, ax = plt.subplots(figsize=(8, 6))

    feature_idx = X_test.columns.get_loc(feature)
    x_data = X_test[feature]
    y_data = shap_values.values[:, feature_idx]
    color_data = y_test

    # 依赖图使用自定义渐变色
    scatter = ax.scatter(x_data, y_data, c=color_data,
                         cmap=color_params['dep_scatter_cmap'],  # 使用自定义渐变色
                         s=color_params['dep_scatter_size'],
                         alpha=color_params['dep_scatter_alpha'])

    # 颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.set_title(target_column, fontsize=10)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(axis='y', length=aesthetic_params['dep_cbar_tick_length'],
                        labelsize=aesthetic_params['tick_label_size'])

    ax.set_xlabel(feature, fontsize=aesthetic_params['ax_label_size'])
    ax.set_ylabel('', fontsize=aesthetic_params['ax_label_size'])

    # 阈值线
    median_val = X_test[feature].median()
    threshold_val = find_knee_point(x_data, y_data)
    ax.axvline(median_val, color=color_params['median_line_color'], linestyle='--', linewidth=1)
    ax.axvline(threshold_val, color=color_params['threshold_line_color'], linestyle=':', linewidth=1.2)

    # 图例
    from matplotlib.lines import Line2D

    line_handles = [
        Line2D([0], [0], color=color_params['median_line_color'], lw=1, linestyle='--',
               label=f'Median: {median_val:.2f}'),
        Line2D([0], [0], color=color_params['threshold_line_color'], lw=1, linestyle=':',
               label=f'Threshold: {threshold_val:.2f}')
    ]
    ax.legend(handles=line_handles, loc='best', fontsize=aesthetic_params['legend_size'])
    ax.tick_params(axis='both', which='major', labelsize=aesthetic_params['tick_label_size'])

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)

    # 保存依赖图
    output_dep_path = rf'C:\Users\lenovo\Desktop\shap_dependency_plot_{feature}.png'
    plt.savefig(output_dep_path, dpi=300, bbox_inches='tight')
    print(f"--> SHAP 依赖图 {feature} 已成功保存到文件: {output_dep_path}")
    plt.close(fig_dep)

# 打印归一化后的特征重要性值
print("\n--> Relative importance of environmental variables influencing soybean yield (normalized):")
print(feature_importance_df.sort_values('importance', ascending=False))

# 验证归一化
print("Sum of normalized importance values:", feature_importance_df['importance'].sum())

print("--> 所有图像保存完成！")