import os
import mysql.connector
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()


def get_connection():

    connection = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

    return connection

def save_prediction(
    tenure,
    monthly_charges,
    total_charges,
    contract,
    internet_service,
    churn_prediction,
    churn_probability
):
    connection = get_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO predictions (
            tenure,
            monthly_charges,
            total_charges,
            contract,
            internet_service,
            churn_prediction,
            churn_probability
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    values = (
        tenure,
        monthly_charges,
        total_charges,
        contract,
        internet_service,
        churn_prediction,
        churn_probability
    )

    cursor.execute(query, values)
    connection.commit()

    cursor.close()
    connection.close()