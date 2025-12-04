# Setup & Deployment Guide

This guide provides detailed instructions for setting up and deploying the Maternal-AI project.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Option A: Google Colab Deployment (Recommended)](#option-a-google-colab-deployment)
3. [Option B: Local Deployment](#option-b-local-deployment)
4. [Troubleshooting](#troubleshooting)
5. [FAQ](#faq)

---

## Prerequisites

### Required Accounts

1. **HuggingFace Account**
   - Sign up at: https://huggingface.co/join
   - Request LLaMA 2 access: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
   - Generate access token: https://huggingface.co/settings/tokens
   - ⏱️ LLaMA 2 approval can take 1-24 hours

2. **Google Account** (for Colab)
   - Required for: https://colab.research.google.com
   - Provides free GPU access

3. **Ngrok Account** (for Colab deployment)
   - Sign up at: https://ngrok.com
   - Get auth token: https://dashboard.ngrok.com/get-started/your-authtoken

---

## Option A: Google Colab Deployment (Recommended)

**Best for**: Beginners, no local GPU, quick deployment

### Step 1: Prepare HuggingFace Access

1. Visit: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it: `maternal-ai-token`
4. Select: `Read` permission
5. Copy the token (you'll need it soon)

### Step 2: Setup Colab Secrets

1. Open any notebook in Google Colab
2. Click the 🔑 key icon on the left sidebar
3. Click "+ Add new secret"
4. Name: `HF_TOKEN`
5. Value: Paste your HuggingFace token
6. Toggle ON "Notebook access"

### Step 3: Run Data Preparation

```python
# Upload GH_data_preparation.ipynb to Colab
# Click: Runtime → Run all
# Wait for completion (~5-10 minutes)
```

**What this does:**
- Creates folder structure in Google Drive
- Generates training dataset
- Saves data to: `/content/drive/MyDrive/maternal_health_project/data/`

### Step 4: Train the Model

```python
# Upload GH_qlora.ipynb to Colab
# IMPORTANT: Enable GPU (Runtime → Change runtime type → T4 GPU)
# Click: Runtime → Run all
# Wait for training (~30-60 minutes)
```

**What this does:**
- Loads LLaMA 2-7B with 4-bit quantization
- Fine-tunes with QLoRA
- Saves model to: `/content/drive/MyDrive/maternal_health_project/models/`

### Step 5: Deploy the App

```python
# Upload GH_Streamlit.ipynb to Colab
# Replace NGROK_TOKEN with your actual token (line 17)
# Run all cells
# Click the ngrok URL that appears
```

**What this does:**
- Loads your fine-tuned model
- Starts Streamlit server
- Creates public URL via ngrok

### ⚠️ Important Colab Notes:

- **Session Timeout**: Colab sessions end after ~12 hours or if idle
- **Save Regularly**: Your work in Drive persists, but Colab runtime resets
- **GPU Quota**: Free tier has daily limits (usually sufficient for this project)

---

## Option B: Local Deployment

**Best for**: Advanced users, have local GPU, development work

### Step 1: System Requirements

**Minimum:**
- GPU: 12GB VRAM (RTX 3060 12GB, RTX 4070, etc.)
- RAM: 16GB system RAM
- Storage: 50GB free space
- OS: Linux/Windows with WSL2

**Recommended:**
- GPU: 24GB VRAM (RTX 3090, RTX 4090, A5000)
- RAM: 32GB system RAM
- Storage: 100GB free SSD

### Step 2: Install CUDA

**Linux:**
```bash
# Ubuntu 22.04 example
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

**Windows:**
- Download from: https://developer.nvidia.com/cuda-downloads
- Follow installer instructions

### Step 3: Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/Maternal-AI.git
cd Maternal-AI

# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Authenticate HuggingFace

```bash
huggingface-cli login
# Paste your token when prompted
```

### Step 5: Prepare Data

```bash
python data_preparation.py
```

**Note**: You may need to modify file paths in the script for local deployment.

### Step 6: Train Model

```bash
# This will take several hours depending on your GPU
python qlora_training.py
```

**Training time estimates:**
- RTX 3090: ~2-3 hours
- RTX 4090: ~1-2 hours
- A100: ~45 minutes

### Step 7: Run Streamlit App

```bash
streamlit run streamlit_app.py
```

Open browser to: http://localhost:8501

---

## Troubleshooting

### Common Issues

#### Issue 1: "CUDA out of memory"

**Solution:**
```python
# In qlora_training.py, reduce batch size:
per_device_train_batch_size = 2  # Instead of 4
gradient_accumulation_steps = 8  # Instead of 4
```

#### Issue 2: "Could not find model"

**Solution:**
- Check LLaMA 2 access is approved
- Verify HuggingFace token has read permissions
- Try re-logging: `huggingface-cli login`

#### Issue 3: Colab disconnects during training

**Solution:**
```javascript
// Run this in browser console to keep Colab active:
function KeepClicking(){
    console.log("Clicking");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(KeepClicking, 60000)
```

#### Issue 4: Ngrok tunnel closed

**Solution:**
- Free tier has 2-hour limit
- Re-run the Streamlit cell to get new URL
- Consider upgrading to ngrok paid plan

#### Issue 5: "ImportError: bitsandbytes"

**Solution:**
```bash
# Linux
pip install bitsandbytes

# Windows
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```

---

## FAQ

### Q: How much does this cost?

**A:** Free options:
- Google Colab: Free tier includes GPU
- HuggingFace: Free account
- Ngrok: Free tier available

### Q: Can I use a different base model?

**A:** Yes! Modify `MODEL_NAME` in the scripts. Popular alternatives:
- `meta-llama/Llama-2-13b-chat-hf` (needs more VRAM)
- `mistralai/Mistral-7B-Instruct-v0.2`
- `google/flan-t5-xl`

### Q: How accurate is the model?

**A:** The model provides general guidance but should never replace professional medical advice. Always use the human evaluation feature to track quality.

### Q: Can I add my own training data?

**A:** Yes! Edit the knowledge base in `data_preparation.py` to include your Q&A pairs.

### Q: How do I update the model after adding more data?

**A:**
1. Run `data_preparation.py` again
2. Run `qlora_training.py` with new data
3. Model will be saved to same location (or specify new OUTPUT_DIR)

### Q: Can I deploy this to a website?

**A:** Yes! Options:
- **Streamlit Cloud**: Free hosting (limited resources)
- **Hugging Face Spaces**: Free GPU inference
- **AWS/GCP/Azure**: Scalable but costs money
- **Railway/Render**: Easy deployment

### Q: Is this HIPAA compliant?

**A:** No. This is a demonstration project and not suitable for handling real patient data without proper security implementations.

---

## Next Steps After Setup

1. **Test the model**: Ask various questions to evaluate responses
2. **Use evaluation features**: Rate responses to track quality
3. **Iterate on training data**: Add more Q&A pairs for better coverage
4. **Share your deployment**: Use ngrok or deploy to cloud
5. **Contribute improvements**: Submit PRs to enhance the project

---

## Getting Help

- **GitHub Issues**: Open an issue in the repository
- **HuggingFace Forums**: https://discuss.huggingface.co
- **Stack Overflow**: Tag with `llama-2`, `streamlit`, `peft`

---

**Happy Deploying! 🚀**
