# Video Understanding Data Pipeline

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Go 1.22+](https://img.shields.io/badge/Go-1.22%2B-00ADD8?logo=go&logoColor=white)](https://golang.org/)
[![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-2.8-017CEE?logo=apache-airflow&logoColor=white)](https://airflow.apache.org/)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20Glue%20%7C%20Lambda-FF9900?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![TwelveLabs](https://img.shields.io/badge/TwelveLabs-Video%20AI-6C3CE1)](https://www.twelvelabs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end data pipeline for **video ingestion, multimodal embedding generation, and semantic search** powered by the [TwelveLabs](https://www.twelvelabs.io/) Embed & Search APIs. Designed to handle large-scale video datasets on AWS with orchestration via Apache Airflow and a high-performance search API written in Go.

---

## Architecture

```
                         ┌──────────────────────────────────────────────────────┐
                         │                   AWS Cloud                          │
                         │                                                      │
  ┌─────────┐  upload   │  ┌──────────┐   trigger   ┌──────────────────────┐  │
  │ Raw      │──────────▶│  │  S3 Raw  │────────────▶│  Airflow             │  │
  │ Videos   │           │  │  Bucket  │             │  (video_ingestion_   │  │
  └─────────┘           │  └──────────┘             │   dag)               │  │
                         │       │                   └──────────┬───────────┘  │
                         │       │ Glue ETL                     │              │
                         │       ▼                              │ TwelveLabs   │
                         │  ┌──────────┐                        │ API calls    │
                         │  │  AWS     │                        ▼              │
                         │  │  Glue    │             ┌──────────────────────┐  │
                         │  │  Job     │             │  TwelveLabs          │  │
                         │  └────┬─────┘             │  Index / Embed API   │  │
                         │       │ metadata           └──────────┬───────────┘  │
                         │       ▼                              │ embeddings    │
                         │  ┌──────────┐                        ▼              │
                         │  │ DynamoDB │◀──────────── ┌──────────────────────┐ │
                         │  │ Metadata │  store        │  Airflow             │ │
                         │  └──────────┘  results     │  (video_embedding_  │ │
                         │                             │   dag)               │ │
                         │  ┌──────────┐              └──────────────────────┘ │
                         │  │  S3      │◀─────────────────────────────────────  │
                         │  │ Embed    │  store embeddings                       │
                         │  │ Bucket   │                                         │
                         │  └──────────┘                                         │
                         │                                                      │
                         │  ┌────────────────────────────────────────────────┐  │
                         │  │         Go Search API (ECS / Lambda)           │  │
                         │  │  GET /search?q=...  →  TwelveLabs Search API   │  │
                         │  │  GET /videos/{id}   →  DynamoDB lookup         │  │
                         │  │  GET /health                                    │  │
                         │  └────────────────────────────────────────────────┘  │
                         └──────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Video AI | TwelveLabs Embed & Search API (Marengo 2.7 / Pegasus 1.2) |
| Orchestration | Apache Airflow 2.8 |
| ETL (Python) | boto3, pydantic, pandas |
| Search API | Go 1.22, net/http, chi router |
| Storage | AWS S3, DynamoDB |
| Batch Processing | AWS Glue (PySpark) |
| Infrastructure | AWS (S3, Glue, Lambda, ECS, IAM) |
| CI/CD | GitHub Actions |

---

## Project Structure

```
video-understanding-pipeline/
├── pipelines/
│   └── airflow/
│       └── dags/
│           ├── video_ingestion_dag.py     # S3 → TwelveLabs index upload
│           └── video_embedding_dag.py    # Embedding generation & storage
├── video_understanding/
│   ├── twelvelabs_client.py              # TwelveLabs API client wrapper
│   └── pipeline.py                       # End-to-end pipeline orchestrator
├── etl/
│   ├── extract.py                        # S3 video extraction & metadata
│   ├── transform.py                      # Normalization & schema validation
│   └── load.py                           # DynamoDB / S3 upsert logic
├── services/
│   └── api/
│       ├── main.go                       # Go HTTP search API
│       └── go.mod                        # Go module definition
├── infra/
│   └── aws/
│       ├── s3_config.py                  # S3 bucket provisioning
│       └── glue_job.py                   # AWS Glue batch job
├── requirements.txt
└── .gitignore
```

---

## Prerequisites

- Python 3.11+
- Go 1.22+
- Apache Airflow 2.8 (local or managed — MWAA / Astronomer)
- AWS account with permissions for S3, DynamoDB, Glue, Lambda, IAM
- [TwelveLabs API key](https://playground.twelvelabs.io/)

---

## Setup

### 1. Clone & install Python dependencies

```bash
git clone https://github.com/your-handle/video-understanding-pipeline.git
cd video-understanding-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your credentials
```

```dotenv
TWELVELABS_API_KEY=tlk_xxxxxxxxxxxxxxxxxxxx
TWELVELABS_INDEX_ID=your-index-id

AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...

S3_RAW_BUCKET=your-raw-videos-bucket
S3_EMBED_BUCKET=your-embeddings-bucket
DYNAMODB_TABLE=video-metadata

AIRFLOW_HOME=/opt/airflow
```

### 3. Start Airflow locally (Docker)

```bash
docker compose up airflow-init
docker compose up -d
# UI available at http://localhost:8080
```

### 4. Build and run the Go search API

```bash
cd services/api
go mod tidy
go run main.go
# API available at http://localhost:8081
```

---

## Key Workflows

### Video Ingestion Pipeline
1. Airflow DAG (`video_ingestion_dag`) polls S3 raw bucket for new `.mp4` / `.mov` files
2. Uploads each video URL to TwelveLabs index via the Upload API
3. Polls indexing status until `ready` or `failed`
4. Writes metadata record to DynamoDB

### Embedding Pipeline
1. Airflow DAG (`video_embedding_dag`) queries DynamoDB for indexed videos without embeddings
2. Calls TwelveLabs Embed API to generate multimodal embeddings
3. Stores embedding vectors as `.npy` files in S3 embed bucket
4. Updates DynamoDB record with embedding S3 path

### Semantic Search (Go API)
- `GET /search?q=<natural language query>&limit=10` — proxies TwelveLabs Search API, returns ranked video clips
- `GET /videos/{id}` — fetches video metadata from DynamoDB
- `GET /health` — liveness probe

---

## Data Model (DynamoDB)

```json
{
  "video_id":      "tlv_abc123",
  "s3_key":        "raw/2024/01/15/interview_clip_01.mp4",
  "index_id":      "idx_xyz789",
  "status":        "embedded",
  "duration_sec":  142.5,
  "file_size_mb":  87.3,
  "created_at":    "2024-01-15T10:23:00Z",
  "indexed_at":    "2024-01-15T10:31:44Z",
  "embedding_s3":  "embeddings/tlv_abc123.npy",
  "tags":          ["interview", "product-demo"],
  "metadata": {
    "source":      "upload-batch-2024-01",
    "resolution":  "1920x1080",
    "fps":         30
  }
}
```

---

## Contributing

Pull requests are welcome. For major changes please open an issue first.

---

## License

[MIT](LICENSE)
