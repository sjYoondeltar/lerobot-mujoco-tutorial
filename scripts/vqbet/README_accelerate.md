# Accelerating VQBeT Model Training with Hugging Face Accelerate

[English](#english) | [한국어](#korean)

<a id="english"></a>
## English Documentation

This document explains how to accelerate VQBeT model training using Hugging Face's Accelerate library.

### 1. Installing Accelerate

First, you need to install the Accelerate library:

```bash
pip install accelerate
```

For additional logging tools, install:

```bash
# For TensorBoard logging
pip install tensorboard

# Or for Weights & Biases logging
pip install wandb
```

### 2. Configuring Accelerate

If you're using Accelerate for the first time, run the following command to generate a configuration file:

```bash
accelerate config
```

This command provides an interactive prompt to configure your distributed training environment. You can set:

- The type of distributed training to use (DDP, DeepSpeed, FSDP, etc.)
- Mixed precision (FP16, BF16, etc.)
- Number of processes
- Logging options

### 3. Training on a Single GPU

The simplest way is to run on a single GPU with the following command:

```bash
python scripts/train_vqbet.py
```

The Accelerator will automatically initialize and detect a single GPU environment.

### 4. Training on Multiple GPUs

To run distributed training across multiple GPUs, use the `accelerate launch` command:

```bash
accelerate launch scripts/train_vqbet.py
```

This command distributes the training across multiple GPUs according to your configuration file.

### 5. Advanced Configuration

You can also specify certain configurations directly in the CLI:

```bash
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=4 scripts/train_vqbet.py
```

### 6. Code Explanation

The `train_vqbet.py` script uses Accelerate as follows:

1. Accelerator initialization:
```python
accelerator = Accelerator(
    mixed_precision='fp16',
    gradient_accumulation_steps=1,
    log_with="tensorboard",
    project_dir=os.path.join(CKPT_DIR, "logs")
)
```

2. Preparing model, optimizer, and dataloader:
```python
policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
```

3. Using accelerator in the backpropagation process:
```python
accelerator.backward(loss)
```

4. Saving the model after training:
```python
unwrapped_model = accelerator.unwrap_model(policy)
unwrapped_model.save_pretrained(ckpt_dir)
```

5. Logging only from the main process:
```python
if accelerator.is_main_process:
    print("Log message")
```

### 7. Checking Results

Training results are saved in the `./ckpt/vqbet_y/logs` directory. You can visualize the results using TensorBoard:

```bash
tensorboard --logdir=./ckpt/vqbet_y/logs
```

### References

- [Hugging Face Accelerate Documentation](https://huggingface.co/docs/accelerate/index)
- [Accelerate GitHub Repository](https://github.com/huggingface/accelerate)

[Back to top](#accelerating-vqbet-model-training-with-hugging-face-accelerate)

---

<a id="korean"></a>
## 한국어 문서

# Hugging Face Accelerate로 VQBeT 모델 훈련 가속화하기

이 문서는 Hugging Face의 Accelerate 라이브러리를 사용하여 VQBeT 모델 훈련을 가속화하는 방법에 대해 설명합니다.

### 1. Accelerate 설치하기

먼저 Accelerate 라이브러리를 설치해야 합니다:

```bash
pip install accelerate
```

추가적으로 로깅 도구를 사용하려면 다음을 설치하세요:

```bash
# TensorBoard 로깅용
pip install tensorboard

# 또는 Weights & Biases 로깅용
pip install wandb
```

### 2. Accelerate 설정하기

처음 Accelerate를 사용한다면, 다음 명령어를 실행하여 설정 파일을 생성하세요:

```bash
accelerate config
```

이 명령어는 대화형 프롬프트를 통해 분산 훈련 환경을 설정할 수 있게 해줍니다. 다음 사항들을 설정할 수 있습니다:

- 사용할 분산 훈련 타입 (DDP, DeepSpeed, FSDP 등)
- 혼합 정밀도 (FP16, BF16 등)
- 프로세스 수
- 로깅 옵션

### 3. 단일 GPU에서 훈련하기

가장 간단한 방법으로, 단일 GPU에서 다음 명령어로 실행할 수 있습니다:

```bash
python scripts/train_vqbet.py
```

코드 내에서 Accelerator가 자동으로 초기화되고 단일 GPU 환경을 감지합니다.

### 4. 다중 GPU에서 훈련하기

여러 GPU에서 분산 훈련을 실행하려면 `accelerate launch` 명령어를 사용하세요:

```bash
accelerate launch scripts/train_vqbet.py
```

이 명령어는 설정 파일에 따라 여러 GPU에 걸쳐 훈련을 분산시킵니다.

### 5. 고급 설정

특정 설정을 CLI에서 직접 지정할 수도 있습니다:

```bash
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=4 scripts/train_vqbet.py
```

### 6. 코드 설명

`train_vqbet.py` 스크립트는 다음과 같이 Accelerate를 사용합니다:

1. Accelerator 초기화:
```python
accelerator = Accelerator(
    mixed_precision='fp16',
    gradient_accumulation_steps=1,
    log_with="tensorboard",
    project_dir=os.path.join(CKPT_DIR, "logs")
)
```

2. 모델, 옵티마이저, 데이터로더 준비:
```python
policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
```

3. 역전파 과정에서 accelerator 사용:
```python
accelerator.backward(loss)
```

4. 훈련 후 모델 저장하기:
```python
unwrapped_model = accelerator.unwrap_model(policy)
unwrapped_model.save_pretrained(ckpt_dir)
```

5. 메인 프로세스에서만 로깅하기:
```python
if accelerator.is_main_process:
    print("로그 메시지")
```

### 7. 결과 확인

훈련 결과는 `./ckpt/vqbet_y/logs` 디렉토리에 저장됩니다. TensorBoard를 사용하여 결과를 시각화할 수 있습니다:

```bash
tensorboard --logdir=./ckpt/vqbet_y/logs
```

### 참고 자료

- [Hugging Face Accelerate 문서](https://huggingface.co/docs/accelerate/index)
- [Accelerate GitHub 저장소](https://github.com/huggingface/accelerate)

[맨 위로](#accelerating-vqbet-model-training-with-hugging-face-accelerate) 