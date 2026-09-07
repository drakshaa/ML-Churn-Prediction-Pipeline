CREATE DATABASE IF NOT EXISTS churn_db;

USE churn_db;

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    tenure INT,
    monthly_charges DECIMAL(10,2),
    total_charges DECIMAL(10,2),

    contract VARCHAR(50),
    internet_service VARCHAR(50),

    churn_prediction INT,
    churn_probability DECIMAL(5,4)
);