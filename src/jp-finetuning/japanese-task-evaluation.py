import os
import numpy as np
import torch
import wandb
import sentencepiece
import argparse
from datasets import load_dataset, load_from_disk
#from wandb.integration.langchain import WandbTracer
from langchain.callbacks.tracers import WandbTracer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, TrainerCallback, pipeline
from langchain import PromptTemplate, HuggingFaceHub, HuggingFacePipeline, LLMChain, OpenAI
from langchain.chains import SequentialChain
from huggingface_hub import HfApi, list_models
from huggingface_hub.inference_api import InferenceApi
from huggingface_hub import login
from prompt_template import get_template
from utils import eval_MARC_ja, eval_JSTS, eval_JNLI, eval_JSQuAD, eval_JCommonsenseQA
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model


#login(os.environ['HUGGINGFACE_TOKEN'])

def parse_args():
    parser = argparse.ArgumentParser(description="upload raw score json file")
    parser.add_argument(
        "--wandb_entity",
        type=str,
        required="True",
        help="wandb entity name",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        required="True",
        help="wandb project name",
    )
    parser.add_argument(
        "--use_local_model",
        type=str,
        required="True",
        help="yes or no",
    )
    parser.add_argument(
        "--adapter_artifact_name",
        type=str,
        required="False",
        help="artifact path of adapter",
    )
    parser.add_argument(
        "--hf_model_name",
        type=str,
        required="True",
        help="if you directly use models on hf, put the model name. if you use local fine-tuned model, put the base model on hf",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required="True",
        help="Used as the displayed name on leaderboard",
    )
    parser.add_argument(
        "--use_artifact_of_dataset",
        type=str,
        required="True",
        help="yes or no",
    )
    parser.add_argument(
        "--prompt_template_type",
        type=str,
        required="True",
        help="prompt_template_typ",
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    eval_category = ['MARC-ja', 'JSTS', 'JNLI', 'JSQuAD', 'JCommonsenseQA']
    args = parse_args()
    with wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args, job_type="eval") as run:
        config = wandb.config
        table_contents = []
        table_contents.append(config["model_name"])
        
        if "rinna" in config.hf_model_name:
            tokenizer = AutoTokenizer.from_pretrained(config.hf_model_name,use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.hf_model_name)
        if "Llama-2" in config.hf_model_name:
            temperature = 1e-9
        else:
            temperature = 0
        template_type = config.prompt_template_type

        if config.use_local_model=="no":
            model = AutoModelForCausalLM.from_pretrained(config.hf_model_name, trust_remote_code=True,torch_dtype=torch.float16)
        else:
            base_llm = AutoModelForCausalLM.from_pretrained(config.hf_model_name, trust_remote_code=True, torch_dtype=torch.float16)
            model_ar = wandb.use_artifact(config.adapter_artifact_name)
            model_path = model_ar.download()
            model = PeftModel.from_pretrained(base_llm, model_path, torch_dtype=torch.float16)
            model = model.merge_and_unload()
            
        #MRAC-ja --------------------------------------------------------
        if config.use_artifact_of_dataset=="yes":
            artifact = run.use_artifact('wandb/LLM_evaluation_Japan/JGLUE-MRAC-ja:v0', type='dataset')
            artifact_dir = artifact.download()
            dataset = load_from_disk(artifact_dir)
        else:
            dataset = load_dataset("shunk031/JGLUE", name=eval_category[0])
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=5, device=0, torch_dtype=torch.bfloat16, temperature=temperature, 
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        llm_chain = LLMChain(llm=llm, prompt=get_template(eval_category[0], template_type), output_key="output")
        marc_ja_score,marc_ja_score_balanced  = eval_MARC_ja(dataset,llm_chain)
        table_contents.append(marc_ja_score)
        table_contents.append(marc_ja_score_balanced)
        
        #JSTS--------------------------------------------------------
        if config.use_artifact_of_dataset=="yes":
            artifact = run.use_artifact('wandb/LLM_evaluation_Japan/JGLUE-JSTS:v0', type='dataset')
            artifact_dir = artifact.download()
            dataset = load_from_disk(artifact_dir)
        else:
            dataset = load_dataset("shunk031/JGLUE", name=eval_category[1])
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=3, device=0, torch_dtype=torch.bfloat16, temperature=temperature,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        llm_chain = LLMChain(llm=llm, prompt=get_template(eval_category[1], template_type), output_key="output")
        jsts_peason, jsts_spearman= eval_JSTS(dataset,llm_chain)
        table_contents.append(jsts_peason)
        table_contents.append(jsts_spearman)
        #JNLI--------------------------------------------------------
        if config.use_artifact_of_dataset=="yes":
            artifact = run.use_artifact('wandb/LLM_evaluation_Japan/JGLUE-JNLI:v0', type='dataset')
            artifact_dir = artifact.download()
            dataset = load_from_disk(artifact_dir)
        else:
            dataset = load_dataset("shunk031/JGLUE", name=eval_category[2])
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=3, device=0, torch_dtype=torch.bfloat16, temperature=temperature,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        llm_chain = LLMChain(llm=llm, prompt=get_template(eval_category[2], template_type), output_key="output")
        jnli_score,jnli_score_balanced = eval_JNLI(dataset,llm_chain)
        table_contents.append(jnli_score)
        table_contents.append(jnli_score_balanced)

        #JSQuAD--------------------------------------------------------
        if config.use_artifact_of_dataset=="yes":
            artifact = run.use_artifact('wandb/LLM_evaluation_Japan/JGLUE-JSQuAD:v0', type='dataset')
            artifact_dir = artifact.download()
            dataset = load_from_disk(artifact_dir)
        else:
            dataset = load_dataset("shunk031/JGLUE", name=eval_category[3])
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=25, device=0, torch_dtype=torch.bfloat16, temperature=temperature,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        llm_chain = LLMChain(llm=llm, prompt=get_template(eval_category[3], template_type), output_key="output")
        JSQuAD_EM, JSQuAD_F1= eval_JSQuAD(dataset,llm_chain)
        
        table_contents.append(JSQuAD_EM)
        table_contents.append(JSQuAD_F1)
 
        #JCommonsenseQA--------------------------------------------------------
        if config.use_artifact_of_dataset=="yes":
            artifact = run.use_artifact('wandb/LLM_evaluation_Japan/JGLUE-JCommonsenseQA:v0', type='dataset')
            artifact_dir = artifact.download()
            dataset = load_from_disk(artifact_dir)
        else:
            dataset = load_dataset("shunk031/JGLUE", name=eval_category[4])
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=5, device=0, torch_dtype=torch.bfloat16, temperature=temperature,
            )
        llm = HuggingFacePipeline(pipeline=pipe)
        llm_chain = LLMChain(llm=llm, prompt=get_template(eval_category[4], template_type), output_key="output")

        JCommonsenseQA = eval_JCommonsenseQA(dataset,llm_chain)
        table_contents.append(JCommonsenseQA)

        #End--------------------------------------------------------
        table = wandb.Table(columns=['model_name ','MARC-ja','MARC-ja-balanced', 'JSTS-pearson',
                                     'JSTS-spearman', 'JNLI','JNLI-balanced', 'JSQuAD-EM', 'JSQuAD-F1', 'JCommonsenseQA'],
                            data=[table_contents])
        table = wandb.Table(columns=['model_name ','MARC-ja','MARC-ja-balanced', 'JSTS-pearson',
                                     'JSTS-spearman', 'JNLI','JNLI-balanced', 'JSQuAD-EM', 'JSQuAD-F1', 'JCommonsenseQA'],
                            data=table.data)
        run.log({'result_table_balanced':table}) 
        run.log_code()

