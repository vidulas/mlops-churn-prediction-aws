# Project Context

Project Name: End-to-End MLOps Pipeline for Customer Churn Prediction on AWS

Status:
Fully deployed on AWS EC2 with Docker, Amazon ECR, and a manual GitHub Actions CI/CD workflow.

Goal:
Build a portfolio-ready MLOps project that trains a customer churn prediction model locally, serves it with FastAPI, containerizes it with Docker, pushes the image to AWS ECR, deploys it on AWS EC2, and automates deployment using GitHub Actions CI/CD.

Completed Work:
- Trained customer churn prediction models using the Telco Customer Churn dataset.
- Compared Logistic Regression and Random Forest.
- Selected Logistic Regression as the final model based on F1-score and ROC-AUC.
- Saved the trained model as `models/churn_model.pkl`.
- Built a FastAPI REST API with `/health` and `/predict` endpoints.
- Containerized the API using Docker.
- Tested the API locally and inside Docker.
- Created an AWS ECR repository named `telco-churn-api`.
- Pushed the Docker image to AWS ECR.
- Created an AWS EC2 instance using Amazon Linux 2023.
- Installed Docker on EC2.
- Pulled the Docker image from ECR and ran the container on EC2.
- Exposed the API on port `8000`.
- Added GitHub Actions secrets for AWS and EC2 deployment.
- Successfully ran the manual GitHub Actions workflow to build, push, and deploy the app.
- Fixed deployment issues including Scikit-learn version mismatch, EC2 SSH access, and Docker permission errors.

Dataset:
Telco Customer Churn dataset with 7,043 customer records. Features include demographics, account information, services, billing values, and the target variable `Churn`.

Target:
`Churn` column with values `Yes` and `No`.

Final Model:
Logistic Regression was selected because it achieved the best F1-score and ROC-AUC among the evaluated models.

Tech Stack:
Python, Pandas, Scikit-learn, FastAPI, Uvicorn, Docker, AWS ECR, AWS EC2, GitHub Actions.

Deployment:
The FastAPI app is deployed as a Docker container on an Amazon Linux 2023 EC2 instance. The Docker image is stored in Amazon ECR under the `telco-churn-api` repository. Deployment is automated through a manually triggered GitHub Actions workflow using `workflow_dispatch`.

Security Note:
Do not include AWS secret values, private keys, account credentials, public IP addresses tied to private infrastructure, or other sensitive information in repository documentation.
