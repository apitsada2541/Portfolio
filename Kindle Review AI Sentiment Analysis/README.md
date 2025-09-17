# Kindle Review AI Sentiment Analysis

## Project Overview
This project implements an advanced AI system for analyzing and summarizing Kindle book reviews to determine customer satisfaction. The system utilizes fine-tuned language model, retrieval-augmented generation (RAG), and multi-agent workflows to provide comprehensive sentiment analysis and rating predictions.

## Key Features
- **Fine-tuned Language Model**: Uses SmolLM2-1.7B-Instruct base model fine-tuned with LoRA (Low-Rank Adaptation) for improved review analysis
- **Multi-Agent Architecture**: LangGraph-based workflow with specialized agents for retrieval, filtering, summarization, and rating prediction
- **Vector Database Integration**: Pinecone vector store for semantic similarity search of reviews
- **Interactive Web Interface**: Streamlit-based application for easy user interaction
- **Flexible Model Selection**: Choose between base and fine-tuned models
- **Advanced RAG Pipeline**: Optional summarization for enhanced context understanding

## System Architecture

### Agent Workflow
1. **Retriever Agent**: Searches for relevant reviews using vector similarity
2. **Filter Agent**: Selects the most relevant review from retrieved results
3. **Summarizer Agent** (Optional): Creates concise summaries of reviews
4. **Rating Agent**: Predicts customer satisfaction ratings (1-5 scale)

### Models Used
- **Base Model**: HuggingFaceTB/SmolLM2-1.7B-Instruct
- **Fine-tuned Model**: pearl41/Pearl_finetuned_smolLM (hosted on Hugging Face)
- **Embedding Model**: OpenAI text-embedding-3-small (768 dimensions)
- **LLM for Processing**: OpenAI GPT-3.5-turbo

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Required API keys:
  - OpenAI API key
  - Pinecone API key
  - Hugging Face Hub token

### Dependencies
```bash
pip install torch transformers datasets
pip install streamlit langchain langchain-openai langchain-pinecone
pip install pinecone-client peft huggingface-hub
pip install pandas scikit-learn
pip install langgraph
```

### Environment Setup
1. Clone or download the project files
2. Set up your API keys in the environment or directly in the code:
   ```python
   os.environ['OPENAI_API_KEY'] = 'your_openai_key'
   os.environ['PINECONE_API_KEY'] = 'your_pinecone_key'
   ```
3. Update Hugging Face login token in the code:
   ```python
   login("your_huggingface_token")
   ```

## File Structure
```
├── agents.py              # Main agent implementation and workflow
├── WebApp.py             # Streamlit web application
├── Unit_Test.py          # Unit testing for model fine-tuning
├── Finetune.ipynb   # Model fine-tuning notebook
├── Complete.ipynb    # Complete workflow demonstration
├── Dockerfile    # Dockerfile for deploy to Docker
├── requirements.txt    # Necessary libraries
└── README.md             # This file
```

## Usage

### Running the Web Application
```bash
streamlit run WebApp.py
```

The web interface provides:
- Text area for entering reviews or queries
- Model selection (base vs. fine-tuned)
- Summarizer toggle option
- Rating-only mode for direct predictions

### Using Individual Components

#### Model Fine-tuning
Run the unit test to verify fine-tuning setup:
```bash
python Unit_Test.py
```

#### Jupyter Notebooks
- `Finetune.ipynb`: Demonstrates the complete fine-tuning process
- `Complete.ipynb`: Shows the full agent workflow implementation


## Configuration Options

### Workflow Modes
1. **Full Pipeline**: Retrieval → Filter → Summarize → Rate
2. **Basic RAG**: Retrieval → Filter → Rate (skip summarization)
3. **Rating Only**: Direct rating prediction without retrieval

### Model Parameters
- **Temperature**: 0.7 (adjustable for creativity vs. consistency)
- **Max New Tokens**: 512
- **Retrieval Count**: 5 similar reviews (k=5)

## Data Requirements
The system expects review data in CSV format with columns:
- `rating`: Customer rating (1-5)
- `reviewText`: Full review text
- `summary`: Review title/summary
- Pre-processed reviews are stored in Pinecone vector database

## Testing
Run unit tests to verify system components:
```bash
python Unit_Test.py
```

This tests:
- Model loading and fine-tuning setup
- LoRA configuration
- Basic training pipeline

## Performance Notes
- **GPU Recommended**: CUDA-enabled GPU significantly improves inference speed
- **Memory Requirements**: ~4GB GPU memory for base model, ~6GB for fine-tuned model
- **API Rate Limits**: Consider OpenAI and Pinecone usage limits for production use

## Troubleshooting

### Common Issues
1. **CUDA Errors**: Ensure PyTorch is installed with CUDA support
2. **API Key Errors**: Verify all API keys are correctly set
3. **Memory Issues**: Reduce batch size or use CPU if GPU memory is insufficient
4. **Pinecone Connection**: Check index name and API key configuration

### Model Loading Issues
If models fail to load:
- Verify Hugging Face authentication
- Check internet connectivity
- Ensure sufficient disk space for model downloads

## Future Enhancements
- Support for additional review platforms
- Real-time model fine-tuning capabilities
- Advanced emotion detection
- Multi-language support
- Batch processing for large datasets
