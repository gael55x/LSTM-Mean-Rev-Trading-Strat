# LSTM Mean Reversion Strategy API

This repository contains a FastAPI application that implements an LSTM-based mean reversion trading strategy. It downloads ticker data via **yfinance**, preprocesses the data with technical indicators, trains an LSTM model for price prediction, and simulates trading signals on test data.

## Table of Contents

1. [Project Structure](#project-structure)  
2. [Endpoints](#endpoints)  
3. [Local Development](#local-development)  
4. [Docker Build & Run](#docker-build--run)  
5. [Pushing to Docker Hub / ECR](#pushing-to-docker-hub--ecr)  
6. [AWS ECS Deployment](#aws-ecs-deployment)  
7. [License](#license)  

---

## Project Structure

```
.
├── Dockerfile
├── README.md
├── app
│   ├── analysis.py
│   ├── config.py
│   ├── data_processing.py
│   ├── main.py
│   ├── model_module.py
│   ├── schema.py
│   └── trading.py
├── original 
│   └── original.py
└── requirements.txt
```

- **app/main.py**  
  Main FastAPI entry point containing all routing logic.

- **app/data_processing.py**  
  Contains functions for data download, feature engineering, and scaling.

- **app/model_module.py**  
  Implements Keras/TensorFlow LSTM model building and training.

- **app/trading.py**  
  Implements backtesting logic and trading simulation.

- **app/analysis.py**  
  Functions for analyzing equity curves, training history, and performance segments.

- **app/config.py**  
  Global configuration variables and state shared across modules.

- **app/schema.py**  
  Pydantic models for request and response validation.

- **Dockerfile**  
  Docker build instructions (using Python 3.9 slim base, installing dependencies, etc.).

- **requirements.txt**  
  Python dependencies required by the application.

---

## Endpoints

The application exposes these main endpoints:

1. **Health Check**
   - **GET /**  
     Returns a simple status message to verify the API is running.
   
2. **Run Strategy**
   - **GET /run-strategy**  
     Downloads data, trains the LSTM model, and simulates trading. Returns performance metrics (MAE, MSE, RMSE, etc.).

3. **Predict**
   - **POST /predict**  
     Expects a JSON body with a sequence of shape `(SEQ_LENGTH x FEATURES)`. Outputs the predicted close price and a decision (Buy/Sell/Hold).

4. **Backtest Details**
   - **GET /backtest-details**  
     Returns detailed backtest results including the final equity curve and summary statistics.

5. **Training Metrics**
   - **GET /training-metrics**  
     Shows the training and validation metrics from the latest LSTM training session.

6. **Performance Periods**
   - **GET /performance-periods**  
     Returns a breakdown of “good” vs. “poor” trading segments from the final equity curve.

7. **Training Analysis**
   - **GET /training-analysis**  
     Provides an analysis of the best/worst training epochs (if available).

### Example Usage

```bash
# Health Check
curl -X GET http://<your-domain-or-ip>:4000/

# Run Strategy
curl -X GET http://<your-domain-or-ip>:4000/run-strategy

# Predict (example body)
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"sequence": [[23456.0, 0.53, 50.3, 0.1, 0.05, 112.1, 145.2], ... 60 total entries ...]}' \
  http://<your-domain-or-ip>:4000/predict
```

---

## Local Development

1. **Clone this repository**  
   
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Create a virtual environment and install dependencies**  
   
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run the FastAPI app locally**  
   
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 4000
   ```
   
   Then, visit `http://localhost:4000/docs` to view the interactive API documentation.

---

## Docker Build & Run

### 1. Build the Docker Image

From the project root (where the Dockerfile is located):

```bash
docker build -t my-trading-strategy:latest .
```

### 2. Run the Container

```bash
docker run -p 4000:4000 --name my-trading-strategy -d my-trading-strategy:latest
```

- Access the API:
  - Health check: `http://localhost:4000/`
  - API docs: `http://localhost:4000/docs`

---

## Pushing to Docker Hub / ECR

### Docker Hub

1. **Log in to Docker Hub**  
   
   ```bash
   docker login
   ```

2. **Tag your image**

   ```bash
   docker tag my-trading-strategy:latest <your-dockerhub-username>/my-trading-strategy:latest
   ```

3. **Push the image**

   ```bash
   docker push <your-dockerhub-username>/my-trading-strategy:latest
   ```

### AWS ECR

1. **Create an ECR Repository** (if not already created):

   ```bash
   aws ecr create-repository --repository-name my-trading-strategy
   ```

2. **Tag your image**

   ```bash
   docker tag my-trading-strategy:latest <aws_account_id>.dkr.ecr.<region>.amazonaws.com/my-trading-strategy:latest
   ```

3. **Log in to ECR and push the image**

   ```bash
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<region>.amazonaws.com
   docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/my-trading-strategy:latest
   ```

---

## AWS ECS Deployment

Follow these steps to deploy the containerized application to AWS ECS (using Fargate):

1. **Create an ECS Cluster**

   - In the AWS Management Console, navigate to ECS.
   - Create a new cluster (choose the "Networking only" option if using Fargate).

2. **Create a Task Definition**

   - Define a new task with Fargate as the launch type.
   - In the container definitions, set:
     - **Image:** Use your pushed ECR image URI (e.g., `<aws_account_id>.dkr.ecr.<region>.amazonaws.com/my-trading-strategy:latest`)
     - **Port Mappings:** Map container port `4000` (or the port you’ve configured) to the host.
     - **CPU and Memory:** Allocate resources as needed (e.g., 512 MB RAM, 0.25 vCPU).
   - Save the task definition.

3. **Create a Service**

   - In your ECS cluster, create a new service based on your task definition.
   - Configure the desired number of tasks.
   - **Networking:** Choose the appropriate VPC, subnets, and security groups. Ensure the security group allows inbound traffic on port `4000`.
   - (Optional) If needed, set up an Application Load Balancer and configure target groups to route traffic to your container.

4. **Access Your Application**

   - Once the service is running (and if using a load balancer, after the target group becomes healthy), access your API using the load balancer’s DNS name or the public IP associated with your service.

---

## License

This project is provided without a specific license by default. Adjust as needed (e.g., MIT, Apache 2.0) if you wish to make it open-source or proprietary.
```
