"""
配置文件
"""

# 数据参数
NUM_CLASSES = 43 # 类别数量
IMG_SIZE = 48  # 图像大小

# 源数据路径
ROOT_DIR_TRAIN = 'D:\\Deep learning DataSet\\GTSRB\\Final_Training\\Images' # 训练数据目录
ROOT_DIR_TEST = 'D:\\Deep learning DataSet\\GTSRB\\Final_Test\\Images' # 测试数据目录
TEST_CSV = 'D:\\Deep learning DataSet\\GTSRB\\Final_Test\\GT-final_test.csv' # 测试集标签

# 转换后数据的路径
X_H5 = 'data_train.h5' # 转换后的训练数据文件
X_TEST_H5 = 'data_test.h5'  # 转换后的测试集数据文件

# 训练参数
LR = 0.01       # 初始学习率 
BATCH_SIZE = 32 # 批次数据大小
EPOCHS = 20      # 训练轮数


MODEL = 'model.h5' # 模型保存路径


