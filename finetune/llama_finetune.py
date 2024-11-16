from unsloth import FastLanguageModel
import torch

max_seq_length = 8196  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",  # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",  # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!

    "unsloth/Llama-3.2-1B-bnb-4bit",  # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
]  # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",  # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

"""We now add LoRA adapters so we only need to update 1 to 10% of all parameters!"""

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

"""<a name="Data"></a>
### Data Prep
We now use the `Llama-3.1` format for conversation style finetunes. We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. But we convert it to HuggingFace's normal multiturn format `("role", "content")` instead of `("from", "value")`/ Llama-3 renders multi turn conversations like below:

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hey there! How are you?<|eot_id|><|start_header_id|>user<|end_header_id|>

I'm great thanks!<|eot_id|>
```

We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3` and more.
"""

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts, }


pass

from datasets import load_dataset

dataset = load_dataset("lilacai/glaive-function-calling-v2-sharegpt", split="train")

"""We now use `standardize_sharegpt` to convert ShareGPT style datasets into HuggingFace's generic format. This changes the dataset from looking like:
```
{"from": "system", "value": "You are an assistant"}
{"from": "human", "value": "What is 2+2?"}
{"from": "gpt", "value": "It's 4."}
```
to
```
{"role": "system", "content": "You are an assistant"}
{"role": "user", "content": "What is 2+2?"}
{"role": "assistant", "content": "It's 4."}
```
"""


def standardize_sharegpt(
        dataset,
        aliases_for_system=["system"],
        aliases_for_user=["user", "human", "input"],
        aliases_for_assistant=["gpt", "assistant", "output"],
        aliases_for_tool=["tool"]  # Added tool aliases
):
    """
    Standardizes ShareGPT and other formats to user/assistant/tool Hugging Face format.

    Adds support for the 'tool' role alongside system, user, and assistant roles.
    """
    import collections
    import itertools

    convos = dataset[:10]["conversations"]
    uniques = collections.defaultdict(list)
    for convo in convos:
        for message in convo:
            for key, value in message.items():
                uniques[key].append(value)

    # Must be only 2 entries
    assert len(uniques.keys()) == 2

    keys = list(uniques.keys())
    length_first = len(set(uniques[keys[0]]))
    length_second = len(set(uniques[keys[1]]))

    if length_first < length_second:
        # Role is assigned to the first element
        role_key = keys[0]
        content_key = keys[1]
    else:
        role_key = keys[1]
        content_key = keys[0]

    # Check roles are in aliases
    all_aliases = set(
        aliases_for_system + aliases_for_user + aliases_for_assistant + aliases_for_tool
    )
    roles = set(uniques[role_key])
    leftover_aliases = (all_aliases | roles) - all_aliases
    if len(leftover_aliases) != 0:
        raise TypeError(
            f"Unsloth: {list(leftover_aliases)} are not in aliases. Please update aliases."
        )

    # Mapping for aliases
    aliases_mapping = {}
    for x in aliases_for_system:    aliases_mapping[x] = "system"
    for x in aliases_for_user:      aliases_mapping[x] = "user"
    for x in aliases_for_assistant: aliases_mapping[x] = "assistant"
    for x in aliases_for_tool:      aliases_mapping[x] = "tool"  # Add mapping for tool

    def _standardize_dataset(examples):
        convos = examples["conversations"]
        all_convos = []
        for convo in convos:
            new_convo = [
                {"role": aliases_mapping[message[role_key]], "content": message[content_key]}
                for message in convo
            ]
            all_convos.append(new_convo)
        return {"conversations": all_convos}

    return dataset.map(_standardize_dataset, batched=True, desc="Standardizing format")


dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched=True, )

"""We look at how the conversations are structured for item 5:"""

print(dataset[5]["conversations"])

"""And we see how the chat template transformed these conversations.

**[Notice]** Llama 3.1 Instruct's default chat template default adds `"Cutting Knowledge Date: December 2023\nToday Date: 26 July 2024"`, so do not be alarmed!
"""

print(dataset[5]["text"])

"""<a name="Train"></a>
### Train the model
Now let's use Huggingface TRL's `SFTTrainer`! More docs here: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support TRL's `DPOTrainer`!
"""

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2056,  # Adjust based on your dataset.
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    dataset_num_proc=4,  # Increase processors for faster preprocessing.
    packing=True,  # Speeds up training for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=500,  # Adjust based on total steps.
        num_train_epochs=3,  # Prefer epochs for full dataset sweeps.
        learning_rate=1e-4,
        fp16=True,
        logging_steps=50,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine_with_restarts",
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=3,
        seed=3407,
        output_dir="outputs/run_1",
        report_to="none",  # Optional: Use "none" if not using monitoring tools.
    ),
)


"""We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs."""

from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

"""We verify masking is actually done:"""

tokenizer.decode(trainer.train_dataset[5]["input_ids"])

space = tokenizer(" ", add_special_tokens=False).input_ids[0]
tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])

"""We can see the System and Instruction prompts are successfully masked!"""

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

"""<a name="Inference"></a>
### Inference
Let's run the model! You can change the instruction and input - leave the output blank!

**[NEW] Try 2x faster inference in a free Colab for Llama-3.1 8b Instruct [here](https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing)**

We use `min_p = 0.1` and `temperature = 1.5`. Read this [Tweet](https://x.com/menhguin/status/1826132708508213629) for more information on why.
"""

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

messages = [
    {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Must add for generation
    return_tensors="pt",
).to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True,
                         temperature=0.1, min_p=0.1)
print(tokenizer.batch_decode(outputs))

"""<a name="Save"></a>
### Saving, loading finetuned models
To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.

**[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!
"""

print("______________________________________")
print("Saving the model adapters")
model.save_pretrained("llama3.2_function_calling_adapter")  # Local saving
tokenizer.save_pretrained("llama3.2_function_calling_adapter")
model.push_to_hub("EmirhanS/llama3.2_function_calling_adapter", token="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")  # Online saving
tokenizer.push_to_hub("EmirhanS/llama3.2_function_calling_adapter", token="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")  # Online saving
print("Model adapters saved")
print("______________________________________")

"""### GGUF / llama.cpp Conversion
To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.

Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
* `q8_0` - Fast conversion. High resource use, but generally acceptable.
* `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
* `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

[**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing)
"""

# Save to 8bit Q8_0
print("______________________________________")
print("Saving the model as Q8")
if True:
    model.save_pretrained_gguf("llama3.2_function_calling_gguf_q8", tokenizer, )
print("Q8 Model saved")
print("______________________________________")

# Save to 16bit GGUF
print("______________________________________")
print("Saving the model as F16")
if True: model.save_pretrained_gguf("llama3.2_function_calling_gguf_f16", tokenizer, quantization_method="f16")
print("F16 Model saved")
print("______________________________________")

# Save to q4_k_m GGUF
print("______________________________________")
print("Saving the model as Q4")
if True: model.save_pretrained_gguf("llama3.2_function_calling_gguf_q4", tokenizer, quantization_method="q4_k_m")
print("Q4 Model saved")
print("______________________________________")

# Save to multiple GGUF options - much faster if you want multiple!
if True:
    print("______________________________________")
    print("Saving the model to hub")
    model.push_to_hub_gguf(
        "EmirhanS/llama3.2_function_calling_gguf",  # Change hf to your username!
        tokenizer,
        quantization_method=["q4_k_m", "q8_0", "q5_k_m", ],
        token="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",  # Get a token at https://huggingface.co/settings/tokens
    )
    print("Model saved to huggingface")
    print("______________________________________")
