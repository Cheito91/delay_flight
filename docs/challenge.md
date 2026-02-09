# LATAM Challenge - Implementation Documentation

## Part I: Model Implementation

### Model Selection

After analyzing the Jupyter notebook (`exploration.ipynb`), I chose **XGBoost with class balancing** as the final model for the following reasons:

#### Performance Comparison

The notebook evaluated two main approaches:

1. **Logistic Regression**
   - Class 0 (No Delay): recall: 0.69, f1-score: 0.67
   - Class 1 (Delay): recall: 0.00, f1-score: 0.00
   - Problem: Cannot predict delays at all (recall = 0)

2. **XGBoost with Class Balancing (`scale_pos_weight`)**
   - Class 0 (No Delay): recall: 0.55, f1-score: 0.67
   - Class 1 (Delay): recall: 0.62, f1-score: 0.40
   - Advantage: Can detect delays with 62% recall

#### Rationale

For a flight delay prediction system, detecting delays (Class 1) is critical for:
- Customer notifications
- Resource allocation
- Schedule adjustments

While the XGBoost model has slightly lower overall accuracy, it successfully predicts delays 62% of the time, making it operationally useful. The Logistic Regression model, despite higher accuracy for non-delays, cannot detect actual delays.

**Key Parameter**: `scale_pos_weight = n_y0 / n_y1` (ratio of non-delayed to delayed flights) balances the class imbalance in the training data.

### Top 10 Features

Based on feature importance analysis in the notebook:

1. `OPERA_Latin American Wings`
2. `MES_7` (July)
3. `MES_10` (October)
4. `OPERA_Grupo LATAM`
5. `MES_12` (December)
6. `TIPOVUELO_I` (International)
7. `MES_4` (April)
8. `MES_11` (November)
9. `OPERA_Sky Airline`
10. `OPERA_Copa Air`

### Bug Fixes

**Line 16 in original `model.py`**:
```python
# Before (incorrect syntax)
Union(Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame)

# After (correct syntax)
Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]
```

Error: `Union` requires square brackets `[]`, not parentheses `()` for type parameters.

### Implementation Details

#### `preprocess(data, target_column=None)`
- Creates one-hot encoded features from `OPERA`, `TIPOVUELO`, and `MES` columns
- Filters to only the top 10 most important features
- If `target_column` is provided:
  - Calculates `min_diff` (difference in minutes between scheduled and actual time)
  - Creates `delay` column (1 if `min_diff` > 15 minutes, else 0)
  - Returns `(features, target)` tuple
- Otherwise returns only `features`

#### `fit(features, target)`
- Calculates class imbalance ratio: `scale = n_y0 / n_y1`
- Initializes XGBoost classifier with:
  - `random_state=1` (reproducibility)
  - `learning_rate=0.01`
  - `scale_pos_weight=scale` (class balancing)
- Trains model on provided features and target

#### `predict(features)`
- Returns list of integer predictions (0 or 1)
- If model not trained, returns all zeros (conservative fallback)

### Test Results

All 4 model tests pass with 92% code coverage:
- test_model_preprocess_for_training
- test_model_preprocess_for_serving
- test_model_fit
- test_model_predict

---

## Part II: API Implementation

### Endpoints

#### `GET /health` (Status: 200)
Health check endpoint
```json
{
  "status": "OK"
}
```

#### `POST /predict` (Status: 200)
Prediction endpoint that accepts flight data and returns delay predictions.

**Request Body**:
```json
{
  "flights": [
    {
      "OPERA": "Aerolineas Argentinas",
      "TIPOVUELO": "N",
      "MES": 3
    }
  ]
}
```

**Response**:
```json
{
  "predict": [0]
}
```

### Validation Rules

Input validation using Pydantic models with custom validators:

1. **MES** (Month): Must be between 1-12
2. **TIPOVUELO** (Flight Type): Must be "N" (National) or "I" (International)
3. **OPERA** (Airline): Must be one of 23 valid airlines in the dataset

Invalid requests return **HTTP 400 Bad Request** with error details.

### Implementation Highlights

- Pydantic Models: Strong typing and automatic validation
- Custom Exception Handler: Converts FastAPI's default 422 validation errors to 400 for consistency with test requirements
- Model Singleton: DelayModel instantiated once at application startup
- Error Handling: Graceful handling of validation and prediction errors

### Dependencies

- fastapi~=0.86.0: Web framework
- pydantic~=1.10.2: Data validation
- uvicorn~=0.15.0: ASGI server
- pandas>=2.0.0: Data manipulation
- numpy>=1.24.0: Numerical computing
- scikit-learn~=1.3.0: ML utilities
- xgboost~=1.7.6: Gradient boosting model

Note: Updated pandas and numpy versions from the original requirements.txt for Python 3.12 compatibility.

### Test Results

All 4 API tests pass:
- test_should_get_predict
- test_should_failed_unkown_column_1 (MES=13, invalid month)
- test_should_failed_unkown_column_2 (TIPOVUELO="O", invalid type)
- test_should_failed_unkown_column_3 (Multiple invalid fields)

---

## Part III: Deployment

### Cloud Provider

Deployed on Google Cloud Platform (GCP) using Cloud Run.

### Deployment Configuration

- Platform: Cloud Run
- Region: us-central1
- Memory: 1Gi
- CPU: 1
- Min instances: 0
- Max instances: 10
- Port: 8080
- Authentication: Allow unauthenticated

### API URL

The deployed API is available at:
```
https://latam-challenge-453188806434.us-central1.run.app
```

This URL has been updated in the Makefile at line 26 as required.

### Deployment Steps

1. Build Docker image using provided Dockerfile
2. Push to GCP Artifact Registry (us-central1-docker.pkg.dev)
3. Deploy to Cloud Run with specified configuration
4. Verify endpoints: /health and /predict

### Stress Test Results

Executed `make stress-test` against deployed API:
- Total requests: 8,282
- Failures: 0 (0.00%)
- Duration: 60 seconds
- Concurrent users: 100
- Throughput: 138.47 req/s
- Average response time: 216ms
- 95th percentile: 370ms
- Report generated: reports/stress-test.html

---

## Part IV: CI/CD

### Workflows Location

CI/CD workflows are located in `.github/workflows/`:
- ci.yml: Continuous Integration
- cd.yml: Continuous Deployment

### Continuous Integration (ci.yml)

Triggers: push and pull_request to all branches

Steps:
1. Checkout code
2. Set up Python 3.12 environment
3. Cache pip dependencies
4. Install dependencies (requirements.txt, requirements-dev.txt, requirements-test.txt)
5. Create data directory and symlink for tests
6. Run model tests: pytest tests/model
7. Run API tests: pytest tests/api
8. Upload coverage reports as artifacts

Purpose: Ensures all tests pass on every push and PR before merging.

### Continuous Deployment (cd.yml)

Triggers: push to main branch only

Steps:
1. Checkout code
2. Authenticate with Google Cloud using service account
3. Configure Docker for Artifact Registry
4. Build Docker image using Cloud Build
5. Deploy to Cloud Run with specified configuration
6. Verify deployment with health check
7. Run smoke tests on deployed API

Required GitHub Secrets:
- GCP_PROJECT_ID: Your GCP project ID
- GCP_SA_KEY: Service account JSON key content
- GCP_REGION: Deployment region (e.g., us-central1)
- GCP_REPOSITORY: Artifact Registry repository name

### Service Account Configuration

A service account with the following roles is required for deployment:

Assigned roles:
- roles/storage.admin
- roles/cloudbuild.builds.builder
- roles/artifactregistry.writer
- roles/iam.serviceAccountUser
- roles/run.admin

---

## Summary

### Completed Tasks

**Part I - Model Implementation**
- Transcribed Jupyter notebook to model.py
- Fixed Union type syntax bug
- Chose XGBoost with class balancing for better delay detection
- Implemented preprocess, fit, and predict methods
- All 4 model tests passing (92% coverage)

**Part II - API Implementation**
- Implemented FastAPI with /health and /predict endpoints
- Added Pydantic validation for OPERA, TIPOVUELO, and MES
- Custom error handler for 400 status codes
- All 4 API tests passing

**Part III - Cloud Deployment**
- Deployed to GCP Cloud Run
- Updated Makefile line 26 with deployment URL
- Stress test completed: 8,282 requests, 0% failures
- API remains deployed and accessible

**Part IV - CI/CD**
- Created .github/workflows/ directory
- Implemented ci.yml for automated testing on all branches
- Implemented cd.yml for automated deployment to main branch
- Configured GCP service account with necessary permissions

### Key Decisions

1. Model Selection: XGBoost with scale_pos_weight chosen over Logistic Regression because it can actually detect delays (62% recall vs 0% recall), which is the primary business requirement.

2. Python Version: Updated dependencies to support Python 3.12 (pandas>=2.0.0, numpy>=1.24.0).

3. Stress Testing: Updated requirements-test.txt to use locust==2.15.1 with compatible flask>=3.0.0, werkzeug>=3.0.0, and jinja2>=3.1.2 to resolve dependency conflicts.

4. Cloud Provider: Selected GCP Cloud Run for serverless deployment with auto-scaling capabilities.

### Test Coverage

- Model tests: 4/4 passing
- API tests: 4/4 passing
- Stress tests: 8,282 requests with 0% failure rate
- Code coverage: 92% in model.py, 88% overall
