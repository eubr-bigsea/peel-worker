from celery import Celery
from dotenv import load_dotenv
import os

load_dotenv()

broker_links = os.getenv('BROKER_LINKS')
broker_links = broker_links.split(';')
broker_list = [f"kafka://{address}" for address in broker_links]
broker_links = ";".join(broker_list)

kafka_topic = os.getenv('BROKER_TOPIC')

db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_user = os.getenv('DB_USERNAME') 
db_password = os.getenv('DB_PASSWORD') 
db_name = os.getenv('DB_NAME') 

db_url = f'db+postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

app_celery = Celery('xai',
             broker=broker_links,
             backend=db_url)

app_celery.conf.task_default_queue = kafka_topic
app_celery.conf.task_track_started = True