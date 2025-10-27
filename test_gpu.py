import torch

# pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
print(torch.__version__)
print(torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用的设备:", device)


# import tensorflow as tf
#
## pip install "tensorflow<2.11"
# # 检查是否可以使用 GPU， cuda<12 required
# print('GPU Available: ', tf.config.list_physical_devices('GPU'))
#
# # 打印当前 TensorFlow 版本
# print("TensorFlow Version: ", tf.__version__)