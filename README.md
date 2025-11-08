# WZQ
本次实验是利用transformer对AG_NEWS数据集进行文本分类的一次实验
其中Tran是包含位置编码的代码，Tran（no position mask）是注释了位置编码的代码
其中AG_NEWS数据集中共有120000条训练数据，7600条测试数据
本次实验选择了其中的1200条作为训练，76条作为测试
<img width="1222" height="453" alt="a2c986162b3171a4c0963fd5f2764858" src="https://github.com/user-attachments/assets/5d498970-380b-4b8d-aac4-9cca1d73bf1c" />

若需要复现可执行以下代码

pip install - r requirements.txt

python Tran.py

tensorboard -- logdir “your code's address”
