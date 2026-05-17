# 环境安装

## 建立conda环境
```bash
conda create -n openvla python=3.10 -y
conda init
source ~/.bashrc #这个命令包含conda init初始化conda环境 出现(base)
conda activate openvla
```
## 安装pytorch文件
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y #安装初始环境
conda clean --all #如果安装失败，清除全部，重新执行上一步安装
```
## 安装Flash Attention 2
```bash
pip install packaging ninja
ninja --version; echo $?
pip install "flash-attn==2.5.5" --no-build-isolation
#(如果直接执行上面的命令一直卡在Building wheel for flash-attn(setup.py)|，可以尝试如下方法，从https://github.com/Dao-AILab/flash-attention/releases下载flash-attn编译好的版本并下载。1. curl -L -O https://ghproxy.net/https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFalse-cp310-cp310-linux_x86_64.whl 2.pip install flash_attn-2.5.5\*.whl)
```
## 安装OpenVLA所需环境
```bash
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .
```
## 安装量化模块
```bash
#安装量化模块bitsandbytes
\# 加 --no-deps不下载几个GB的 nvidia- 依赖包
pip install bitsandbytes==0.42.0 --no-deps
pip install scipy
```
# 下载大模型
```bash
export HF_ENDPOINT=https://hf-mirror.com #使用huggingface镜像
huggingface-cli download --resume-download openvla/openvla-7b-finetuned-libero-spatial --local-dir /root/autodl-tmp/openvla-main/MODEL/[vlarl-7b-finetuned-libero-spatial](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial)
```
# #下载微调数据集
```bash
huggingface-cli download --repo-type dataset openvla/modified_libero_rlds --include "libero_10_no_noops/\*" --local-dir /root/autodl-tmp/LIBERO/libero_10_no_noops --local-dir-use-symlinks False
```
# 进行LoRA微调
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \\
\--vla_path /root/autodl-tmp/openvla-main/model/openvla-7b \\ #模型路径
\--data_root_dir /root/autodl-tmp/LIBERO/libero_10_no_noops \\ #数据集路径
\--dataset_name libero_10_no_noops \\ #数据集名称
\--run_root_dir /root/autodl-tmp/openvla-main/output-p \\ #输出权重路径
\--adapter_tmp_dir /root/autodl-tmp/openvla-main/temperal-weights \\ #优化器路径
\--lora_rank 32 \\
\--batch_size 8 \\
\--grad_accumulation_steps 2 \\
\--learning_rate 5e-4 \\
\--image_aug True\\
\--wandb_project my_vla_project \\
\--wandb_entity my_user_name \\
\--save_steps 500
```
# 推理

## 测试基准LIBERO安装
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e . #安装可编辑环境
cd openvla
pip install -r experiments/robot/libero/libero_requirements.txt #进一步安装所需环境
pip install "numpy<2.0.0"
pip install "opencv-python<4.11" #根据报错补全数据包
```
## 运行推理文件
```bash
python experiments/robot/libero/run_libero_eval.py \\
\--model_family openvla \\
\--pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \\ #加载训练好的模型
\--task_suite_name libero_spatial \\ #任务定义
\--center_crop True
```
# 改进——加入PPO强化学习策略

## 执行训练
```bash
python ppo.py \\
\--pretrained_checkpoint /root/autodl-tmp/openvla-main/model/openvla-7b \\
\--data_root_dir /root/autodl-tmp/LIBERO/libero_10_no_noops \\
\--dataset_name libero_10_no_noops \\
\--run_root_dir /root/autodl-tmp/openvla-main/output-p \\
\--adapter_tmp_dir /root/autodl-tmp/openvla-main/temperal-weights \\
\--lora_rank 32 \\
\--per_device_train_batch_size 8 \\
\--gradient_accumulation_steps 2 \\
\--learning_rate 5e-4 \\
\--image_aug True \\
\--wandb_project my_vla_project \\
\--wandb_entity my_user_name \\
\--save_freq 500
```
## 推理
```bash
python experiments/robot/libero/run_libero_eval.py \\
\--model_family openvla \\
\--pretrained_checkpoint openvla/openvla-7b-ppo-libero-spatial \\ #加载训练好的模型
\--task_suite_name libero_spatial \\ #任务定义
\--center_crop True
```