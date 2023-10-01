# Wandb x NVIDIA Healthcare AI handson lab
This repository is for Wandb x NVIDIA Healthcare AI handson lab. 
This handson covers basic parameter efficient hyperparameter tuning (PEFT) flow with LoRa and protein language model finetuning.

## Environment setting
Select a way of environment setting.
If you attend our class, please see option 1.
If you use codes in this repository on your own environment, please follow option 2.
### option 1 (we use this option for class) Use provided runai environment
1. Create project on runai
2. Move on to terminal and run the following code
```
gh repo clone olachinkei/hc_ai_handson_lab
pip install -r requirements.txt
```


### option 2: Use your own environment
1. set environment variable of `WANDB_API_KEY`
2. build docker container
```
docker-compose up -d
``
