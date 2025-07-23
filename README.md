<picture>
  <source media="(prefers-color-scheme: dark)" srcset="raptor_dark.png">
  <img alt="Shows an illustrated sun in light color mode and a moon with stars in dark color mode." src="raptor.jpg">
</picture>

## RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

**RAPTOR** introduces a novel approach to retrieval-augmented language models by constructing a recursive tree structure from documents. This allows for more efficient and context-aware information retrieval across large texts, addressing common limitations in traditional language models. 



For detailed methodologies and implementations, refer to the original paper:

- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)

[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm.svg)](https://huggingface.co/papers/2401.18059)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/raptor-recursive-abstractive-processing-for/question-answering-on-quality)](https://paperswithcode.com/sota/question-answering-on-quality?p=raptor-recursive-abstractive-processing-for)

## Installation

Before using RAPTOR, ensure Python 3.11 is installed. Clone the RAPTOR repository and install the dependencies with [uv](https://github.com/astral-sh/uv):

```bash

# uv (fast, lock‑file aware)
git clone https://github.com/th-yong/raptor.git
cd raptor
uv sync          # uses the existing pyproject.toml and uv.lock
```

## Basic Usage (Azure OpenAI)

This fork is streamlined for **Azure OpenAI** deployments. The minimal end‑to‑end example is shown below—simply swap in your own endpoint, deployment names, and keys.

```python

# 1️⃣ Azure client managers
azure_emb_client = AzureAIClientManager(
    endpoint="https://<your-resource-name>.cognitiveservices.azure.com/",
    api_key="<your-azure-api-key>",
    deployment="text-embedding-3-large"
)
emb_model = AzureEmbeddingModel(azure_emb_client)

azure_chat_client = AzureAIClientManager(
    endpoint="https://<your-resource-name>.cognitiveservices.azure.com/",
    api_key="<your-azure-api-key>",
    deployment="gpt-4o"
)
sum_model = AzureSummarizationModel(azure_chat_client)
qa_model  = AzureQAModel(azure_chat_client)

# 2️⃣ Configure RAPTOR
cfg = RetrievalAugmentationConfig(
    embedding_model=emb_model,
    summarization_model=sum_model,
    qa_model=qa_model
)

# 3️⃣ Index documents
with open("demo/sample.txt") as f:
    text = f.read()

RA = RetrievalAugmentation(config=cfg)
RA.add_documents(text)

# 4️⃣ Ask a question
print(RA.answer_question("How did Cinderella reach her happy ending?"))
```

For a richer walkthrough—including tree inspection and save/load—open **`demo_custom.ipynb`**.

## Running chunk_and_run.py

The `chunk_and_run.py` script provides a complete example of document ingestion and Q&A using RAPTOR with semantic chunking. To run it:

1. **Set up environment variables** by creating a `.env` file in the project root:
   ```
   AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.cognitiveservices.azure.com/
   AZURE_OPENAI_KEY=<your-azure-api-key>
   ```

2. **Run the script**:
   ```bash
   uv run python chunk_and_run.py
   ```

The script will:
- Load the document from `demo/현대해상3(퇴직연금)상품약관.txt`
- Split it into semantic chunks using `ClusterSemanticTableTextSplitter`
- Build a RAPTOR tree structure
- Answer the question "이율적용형 이율 비율이란?"

You can modify the document path and question in the `main()` function to test with your own content.

### Interactive Notebook Demo

If you prefer an executable walkthrough, open **`demo_custom.ipynb`** in Jupyter Lab or VS Code.  
The notebook reproduces the code samples above and shows how to register your own summarization, QA, and embedding models end‑to‑end.

Note: More examples and ways to configure RAPTOR are forthcoming. Advanced usage and additional features will be provided in the documentation and repository updates.
