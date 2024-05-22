# Two-sentence Horror Storyteller

This is inspired by the subreddit [r/twosentencehorror on Reddit](https://www.reddit.com/r/twosentencehorror).

**Give me a sentence, and I give you a horrific (?) two-sentence story.**

This a simple web chat app that always gives one sentence, given the user's one input sentence, to form a two-sentence horror story. The backbone of the app is [Llama 3 8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), finetuned with 990 all-time top posts on [r/twosentencehorror](https://www.reddit.com/r/twosentencehorror) crawled using PRAW.

Why only 990? In 2023, [Reddit changes its API policies](https://en.wikipedia.org/wiki/2023_Reddit_API_controversy) to limit the amount of posts one user can crawl for a period of time.

Since the training data is so small, the model might not work as well as I intended, horror side. If there is enough time and effort left in me, I'll try to improve the model, but no guarantees.

## Technical overview
The app has two parts:
- Frontend: Uses `streamlit`.
- Backend: Uses `fastapi`.
    - Backbone model: Llama 3 8B-Instruct (https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), finetuned with 990 all-time top posts on [r/twosentencehorror](https://www.reddit.com/r/twosentencehorror) crawled using PRAW, for 3 epochs.
    - Finetuning: Using HuggingFace's `peft` and `bitsandbytes` to finetune the model with QLoRA (LoRA with 4-bit quantization).
    - Inference: Using `bitsandbytes` for `LLM.int8()` quantization.

## Setup
Recommended to run on a machine with an NVIDIA GPU with at least 24GB VRAM (RTX 3090 and up).
### Manually
- Install requirements:
```bash
pip install -r requirements.txt
```
- Log in to HuggingFace using an account with access to the Llama 3 model repositories:
```bash
huggingface-cli login
```
- Download the model adapter (if you want to use my finetuned version)
```bash
gdown 1zeN4nY3Q7O16mRdiL6vTOT-me6TTBX3a -O backend/qlora-3e.zip
cd backend
unzip qlora-3e.zip
```
- Use `tmux` or a seperate shell to run the backend application (I recommend `tmux`):
```bash
cd backend/application
fastapi run app.py
```
You might have to wait for the model to be downloaded if you have not done it already.
- Use `tmux` or a seperate shell to run the frontend application (I recommend `tmux`):
```bash
cd frontend
streamlit run app.py
```
- Access the app at `http://localhost:8501`.
### With Docker (experimental)
- Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Build the image using the Dockerfile
```bash
docker build . -t tsh:latest
```
- Get a container
```bash
docker run -dt tsh:latest
```

- Get into the container to log in to HuggingFace and run the apps (for steps, see the manual install section. Note: no need to download `qlora-3e.zip` because the Dockerfile has already done that)

```bash
docker exec -it <container_name> /opt/nvidia/nvidia_entrypoint.sh
```

### Notes:
- The app is tested on Python 3.10 and Python 3.11. Should also work with 3.12, but there are syntaxes which do not work in 3.9 or under.
- For other uses in training or inference, change the YAML config files (`backend/infer-configs.yaml` for inference and `backend/application/model/training-configs.yaml` for training) accordingly.

## Usage

Coming soon

## References
- QLoRA
```bibtex
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```
- LLM.int8()
```bibtex
@misc{dettmers2022llmint8,
      title={LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale},
      author={Tim Dettmers and Mike Lewis and Younes Belkada and Luke Zettlemoyer},
      year={2022},
      eprint={2208.07339},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
- Llama 3: https://ai.meta.com/blog/meta-llama-3/