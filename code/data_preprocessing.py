import sys
import os
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QMessageBox, QProgressBar
from PyQt5.QtCore import Qt

class MergeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('CSV 合并与插值工具')
        self.setGeometry(100, 100, 400, 250)

        layout = QVBoxLayout()

        self.label1 = QLabel('请选择第一个文件作为基准网格', self)
        layout.addWidget(self.label1)

        self.btn_select_first_file = QPushButton('选择基准文件', self)
        self.btn_select_first_file.clicked.connect(self.select_first_file)
        layout.addWidget(self.btn_select_first_file)

        self.label2 = QLabel('请选择包含其他文件的文件夹', self)
        layout.addWidget(self.label2)

        self.btn_select_input_folder = QPushButton('选择输入文件夹', self)
        self.btn_select_input_folder.clicked.connect(self.select_input_folder)
        layout.addWidget(self.btn_select_input_folder)

        self.label3 = QLabel('请选择输出文件的路径', self)
        layout.addWidget(self.label3)

        self.btn_select_output_file = QPushButton('选择输出文件', self)
        self.btn_select_output_file.clicked.connect(self.select_output_file)
        layout.addWidget(self.btn_select_output_file)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)

        self.btn_merge = QPushButton('合并与插值', self)
        self.btn_merge.clicked.connect(self.merge_files)
        layout.addWidget(self.btn_merge)

        self.setLayout(layout)

        self.first_file = None
        self.input_folder = None
        self.output_file = None

    def select_first_file(self):
        options = QFileDialog.Options()
        self.first_file, _ = QFileDialog.getOpenFileName(self, "选择基准文件", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if self.first_file:
            self.label1.setText(f'已选择基准文件: {self.first_file}')

    def select_input_folder(self):
        self.input_folder = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if self.input_folder:
            self.label2.setText(f'已选择输入文件夹: {self.input_folder}')

    def select_output_file(self):
        options = QFileDialog.Options()
        self.output_file, _ = QFileDialog.getSaveFileName(self, "选择输出文件", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if self.output_file:
            self.label3.setText(f'已选择输出文件: {self.output_file}')

    def merge_files(self):
        if not self.first_file or not self.input_folder or not self.output_file:
            QMessageBox.warning(self, '警告', '请先选择基准文件、输入文件夹和输出文件路径')
            return

        try:
            # 读取基准文件，尝试使用 GBK 编码
            first_df = self.read_csv_with_encoding(self.first_file)
            lon_ref = first_df['Longitude'].values
            lat_ref = first_df['Latitude'].values

            # 创建一个包含所有特征值的 DataFrame
            combined_df = pd.DataFrame({
                'Longitude': lon_ref,
                'Latitude': lat_ref,
                'Yield': first_df['Yield'].values  # 添加基准文件的 Yield 列
            })

            # 遍历输入文件夹中的文件
            files = sorted(os.listdir(self.input_folder))
            total_files = len(files)
            self.progress_bar.setMaximum(total_files)
            self.progress_bar.setValue(0)

            for i, file in enumerate(files):
                file_path = os.path.join(self.input_folder, file)
                print(f"正在处理文件: {file_path}")

                # 读取文件，尝试使用不同编码
                df = self.read_csv_with_encoding(file_path)
                if df is None:
                    continue

                if 'Longitude' not in df.columns or 'Latitude' not in df.columns:
                    print(f"文件 {file} 缺少 Longitude 或 Latitude 列，跳过处理")
                    continue

                # 获取第三列的名称（属性名）
                attribute_name = df.columns[2]  # 假设第三列是属性列

                # 如果当前文件是基准文件，直接添加特征值
                if np.array_equal(df['Longitude'].values, lon_ref) and np.array_equal(df['Latitude'].values, lat_ref):
                    combined_df[attribute_name] = df[attribute_name].values
                    continue

                # 否则，进行插值
                lon_current = df['Longitude'].values
                lat_current = df['Latitude'].values
                feature_current = df[attribute_name].values

                # 使用最近邻插值方法，避免空值
                feature_interpolated = griddata(
                    (lon_current, lat_current),
                    feature_current,
                    (lon_ref, lat_ref),
                    method='nearest'
                )

                # 将插值后的特征值添加到全局 DataFrame
                combined_df[attribute_name] = feature_interpolated

                # 更新进度条
                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()  # 确保界面更新

            # 保存合并结果
            combined_df.to_csv(self.output_file, index=False, encoding='utf-8')
            QMessageBox.information(self, '成功', f'合并完成。结果保存到: {self.output_file}')

        except Exception as e:
            QMessageBox.critical(self, '错误', f'发生错误: {str(e)}')

    def read_csv_with_encoding(self, file_path):
        """
        尝试使用不同编码读取 CSV 文件
        """
        encodings = ['gbk', 'utf-8', 'ISO-8859-1']  # 优先尝试 GBK
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue
        print(f"文件 {file_path} 无法使用以下编码读取: {encodings}")
        return None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MergeApp()
    ex.show()
    sys.exit(app.exec_())