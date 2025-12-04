# Maternal-AI 🤰

An AI-powered maternal health assistant utilizing fine-tuned LLaMA 2 language model to provide guidance and support for expectant mothers.

## 📋 Project Overview

Maternal-AI is a comprehensive machine learning project that leverages QLoRA (Quantized Low-Rank Adaptation) fine-tuning to create a specialized AI assistant for maternal health queries. Built on Meta's LLaMA 2-7B model, the system provides accessible, empathetic, and informative responses to pregnancy-related questions through an intuitive Streamlit web interface.

## 🗂️ Project Structure

```
Maternal-AI/
├── GH_data_preparation.ipynb    # Original Jupyter notebook for data prep
├── GH_qlora.ipynb               # Original QLoRA training notebook
├── GH_Streamlit.ipynb           # Original Streamlit app notebook
├── data_preparation.py          # Converted Python script for data prep
├── qlora_training.py            # Converted QLoRA training script
├── streamlit_app.py             # Converted Streamlit app script
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── README.md                    # Project documentation
├── GITHUB_UPLOAD_GUIDE.md       # Detailed GitHub upload instructions
└── QUICK_REFERENCE.md           # Quick command reference
```

## 🚀 Features

- **Maternal Health Knowledge Base**: Custom-curated dataset of pregnancy-related Q&A
- **QLoRA Fine-tuning**: Memory-efficient 4-bit quantized training on LLaMA 2-7B
- **Interactive Web Interface**: Professional Streamlit application with chat history
- **Human Evaluation System**: Built-in feedback mechanism for response quality
- **Google Drive Integration**: Seamless model and data persistence
- **Emergency Protocol**: Safety features for medical emergencies

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (T4 or better recommended for training)
- Google Colab account (for original notebooks)
- HuggingFace account with LLaMA 2 access
- Ngrok account (for Streamlit deployment)

### Local Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Maternal-AI.git
cd Maternal-AI
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up HuggingFace authentication:**
```bash
huggingface-cli login
```
Enter your HuggingFace token when prompted.

## 💻 Usage

### Option 1: Using Jupyter Notebooks (Recommended for Google Colab)

1. **Data Preparation:**
   - Open `GH_data_preparation.ipynb` in Google Colab
   - Follow the cell-by-cell instructions
   - Creates training dataset in Google Drive

2. **Model Training:**
   - Open `GH_qlora.ipynb` in Google Colab
   - Ensure GPU runtime is enabled (Runtime → Change runtime type → GPU)
   - Run all cells to fine-tune LLaMA 2

3. **Deploy Application:**
   - Open `GH_Streamlit.ipynb` in Google Colab
   - Set your Ngrok auth token
   - Run to launch public web interface

### Option 2: Using Python Scripts (For Local Deployment)

1. **Data Preparation:**
```bash
python data_preparation.py
```

2. **Model Training:**
```bash
python qlora_training.py
```

3. **Run Application:**
```bash
streamlit run streamlit_app.py
```

## 🛠️ Technologies Used

- **LLaMA 2-7B-chat**: Base language model from Meta AI
- **Transformers**: HuggingFace transformers library
- **PEFT**: Parameter-Efficient Fine-Tuning with LoRA
- **BitsAndBytes**: 4-bit model quantization
- **TRL**: Supervised fine-tuning trainer
- **Streamlit**: Web application framework
- **PyTorch**: Deep learning framework
- **Ngrok**: Public URL tunneling for Colab deployment

## 📊 Model Details

- **Base Model**: `meta-llama/Llama-2-7b-chat-hf`
- **Fine-tuning Method**: QLoRA (4-bit NF4 quantization)
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Target modules: q_proj, k_proj, v_proj, o_proj
  - Dropout: 0.05
- **Training Data**: Custom maternal health Q&A dataset
- **Max Sequence Length**: 512 tokens
- **Task**: Conversational question-answering for maternal health

## 📊 Model Details

- **Base Model**: `meta-llama/Llama-2-7b-chat-hf`
- **Fine-tuning Method**: QLoRA (4-bit NF4 quantization)
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Target modules: q_proj, k_proj, v_proj, o_proj
  - Dropout: 0.05
- **Training Data**: Custom maternal health Q&A dataset
- **Max Sequence Length**: 512 tokens
- **Task**: Conversational question-answering for maternal health

## 🎯 Training Configuration

- **Training Framework**: Supervised Fine-Tuning (SFT) with TRL
- **Optimizer**: Paged AdamW 32-bit
- **Learning Rate**: 2e-4 with cosine scheduler
- **Batch Size**: 4 per device with gradient accumulation
- **Precision**: FP16 mixed precision
- **Hardware**: T4 GPU (15GB VRAM)

## 🔒 Safety & Disclaimer

This AI assistant is designed to provide **general information and support only**.

⚠️ **Important**: 
- This is **NOT a substitute for professional medical advice**
- Always consult qualified healthcare providers for medical decisions
- In case of emergency, call your local emergency services immediately
- The app includes built-in emergency protocol detection

## 📸 Features Showcase

The Streamlit application includes:
- 💬 **Chat Interface**: Natural conversation flow with chat history
- 📊 **Human Evaluation**: Rate responses for relevance, empathy, and quality
- 🎨 **Professional UI**: Clean, maternal-health themed design
- 📝 **Conversation Logging**: Track interaction history
- 🚨 **Emergency Detection**: Identifies urgent medical situations

## 🔧 Configuration

### For Google Colab Users:

1. **HuggingFace Token**: Store in Colab Secrets as `HF_TOKEN`
2. **Ngrok Token**: Required for public URL (set in notebook)
3. **Google Drive**: Automatically mounts for persistence

### For Local Users:

1. **Model Path**: Update `BEST_MODEL_PATH` in `streamlit_app.py`
2. **Base Model**: Ensure you have LLaMA 2 access approval
3. **GPU**: CUDA-compatible GPU strongly recommended

## 📈 Project Workflow

```
1. Data Preparation
   ├── Create knowledge base
   ├── Format for instruction tuning
   └── Save to Google Drive

2. Model Training
   ├── Load LLaMA 2-7B with 4-bit quantization
   ├── Apply LoRA adapters
   ├── Fine-tune with SFT
   └── Save trained model

3. Deployment
   ├── Load fine-tuned model
   ├── Create Streamlit interface
   ├── Deploy with Ngrok (Colab) or locally
   └── Enable human evaluation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution:
- Expand the knowledge base with more Q&A pairs
- Improve response quality evaluation metrics
- Add multilingual support
- Enhance UI/UX design
- Add more comprehensive testing

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - [Your GitHub Profile]

## 🙏 Acknowledgments

- **Meta AI** for LLaMA 2
- **HuggingFace** for transformers and PEFT libraries
- **Tim Dettmers** for bitsandbytes quantization
- **Streamlit** for the amazing web framework
- **Google Colab** for free GPU access

## 📧 Contact

For questions, suggestions, or feedback:
- Email: [your-email@example.com]
- GitHub Issues: [Link to your repo issues]

## 📚 Resources

- [LLaMA 2 Paper](https://arxiv.org/abs/2307.09288)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)
- [Streamlit Documentation](https://docs.streamlit.io)

## 🐛 Known Issues

- Colab sessions timeout after inactivity - save work frequently
- Large model downloads require stable internet connection
- Ngrok free tier has session limits

## 🔮 Future Enhancements

- [ ] Add RAG (Retrieval Augmented Generation) for medical references
- [ ] Implement conversation summarization
- [ ] Add voice input/output capabilities
- [ ] Create mobile-friendly version
- [ ] Integrate with health tracking APIs
- [ ] Multi-model comparison interface
- [ ] Automated evaluation metrics

---

**Note**: Remember to:
1. Replace placeholder text (GitHub username, email, etc.) with your actual information
2. Request access to LLaMA 2 on HuggingFace before training
3. Never commit API keys or tokens to the repository
4. Review all outputs from the AI for medical accuracy

**Made with ❤️ for maternal health**
