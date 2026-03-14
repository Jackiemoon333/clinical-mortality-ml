import sqlite3
import pandas as pd
from config import EICU_DB


def get_connection():
    """
    Create a connection to the eICU SQLite database.
    """
    conn = sqlite3.connect(EICU_DB)
    return conn


def run_query(query: str) -> pd.DataFrame:
    """
    Run a SQL query and return the result as a pandas DataFrame.
    """
    conn = get_connection()
    
    df = pd.read_sql(query, conn)
    
    conn.close()
    
    return df