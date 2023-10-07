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
docker compose up -d --build
```

If you want to run python file on this environment, Run the following code
```
docker exec -it <container name> python xxx.py
```
If you want to open jupyterlab, Run the following code.
```
docker exec -it hc_ai_handson_lab /bin/bash
```
You can find jupyter lab tocken with `jupyter lab list`

If you want to learn the basics of environment setup in Japanese, please refer [自前GPUをDeep Learning開発用にセットアップしてみた](https://kkamata.com/%e8%87%aa%e5%89%8dgpu%e3%82%92deep-learning%e9%96%8b%e7%99%ba%e7%94%a8%e3%81%ab%e3%82%bb%e3%83%83%e3%83%88%e3%82%a2%e3%83%83%e3%83%97%e3%81%97%e3%81%a6%e3%81%bf%e3%81%9f/).
