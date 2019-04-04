"""
配置文件

"""
# 数据参数
NUM_CLASSES = 43 # 类别数量
IMG_SIZE = 48  # 图像大小

# 文件路径参数
ROOT_DIR_TRAIN = 'D:\\GTSRB\\Final_Training\\Images' # 训练数据目录
ROOT_DIR_TEST = 'D:\\GTSRB\\Final_Test\\Images' # 测试数据目录

TEST_CSV = 'D:\\GTSRB\\Final_Test\\GT-final_test.csv' # 测试集标签

X_H5 = 'X.h5' # 转换后的训练数据文件
X_TEST_H5 = 'X_test.h5'  # 转换后的测试集数据文件

# 训练参数
LR = 0.01 # 初始学习率 
BATCH_SIZE = 32 # 批次数据大小
EPOCHS = 2 # 训练轮数

MODEL = 'model.h5' # 模型保存路径


