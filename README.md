# MLOps Churn Prediction on AWS

FastAPI service for predicting customer churn from the Telco Customer Churn dataset.

## Local Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train the model if `models/churn_model.pkl` is missing:

```powershell
python src\train.py
```

Run the FastAPI app locally:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open the API docs:

```text
http://localhost:8000/docs
```

Check app health:

```powershell
curl http://localhost:8000/health
```

Send a prediction request:

```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1026.0
  }'
```

## Docker

Build the Docker image:

```powershell
docker build -t telco-churn-api .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 telco-churn-api
```

Then visit:

```text
http://localhost:8000/docs
```

## AWS Deployment

Set these values for your AWS account and preferred region:

```powershell
$AWS_REGION = "us-east-1"
$AWS_ACCOUNT_ID = "123456789012"
$ECR_REPOSITORY = "telco-churn-api"
$IMAGE_TAG = "latest"
$ECR_URI = "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY"
```

Create an AWS ECR repository:

```powershell
aws ecr create-repository `
  --repository-name $ECR_REPOSITORY `
  --region $AWS_REGION
```

Authenticate Docker with ECR:

```powershell
aws ecr get-login-password --region $AWS_REGION | docker login `
  --username AWS `
  --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
```

Build the Docker image locally:

```powershell
docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} .
```

Tag the image for ECR:

```powershell
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${ECR_URI}:${IMAGE_TAG}
```

Push the image to ECR:

```powershell
docker push ${ECR_URI}:${IMAGE_TAG}
```

Connect to your EC2 instance:

```powershell
ssh -i path\to\your-key.pem ec2-user@your-ec2-public-ip
```

Install and start Docker on Amazon Linux EC2 if needed:

```bash
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user
```

Log out and reconnect after adding `ec2-user` to the Docker group.

Authenticate Docker with ECR from EC2:

```bash
aws ecr get-login-password --region us-east-1 | docker login \
  --username AWS \
  --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
```

Pull the image on EC2:

```bash
docker pull 123456789012.dkr.ecr.us-east-1.amazonaws.com/telco-churn-api:latest
```

Run the API container on EC2:

```bash
docker run -d \
  --name telco-churn-api \
  -p 8000:8000 \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/telco-churn-api:latest
```

Open the EC2 security group port:

1. Go to the EC2 console.
2. Select the EC2 instance.
3. Open the attached security group.
4. Add an inbound rule:
   - Type: Custom TCP
   - Port range: 8000
   - Source: your IP address for safer access, or `0.0.0.0/0` for public testing
5. Save the rule.

After the port is open, visit:

```text
http://your-ec2-public-ip:8000/docs
```
