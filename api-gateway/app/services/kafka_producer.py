from confluent_kafka import Producer
import json
import logging

logger = logging.getLogger(__name__)

conf = {'bootstrap.servers': 'kafka:9092'}
producer = Producer(conf)

def produce_to_kafka(topic: str, message: dict):
    try:
        producer.produce(topic, key=str(message.get('id', 'default')), value=json.dumps(message))
        producer.flush()
    except Exception as e:
        logger.error(f"Kafka error: {e}")
        raise
