#!/usr/bin/env python3
"""
Kafka setup script for FraudGuard 360.
Creates necessary topics for the fraud detection pipeline.
"""

import time
import logging
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
from shared.utils import KAFKA_TOPICS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_kafka_topics():
    """Create Kafka topics required for the fraud detection system."""
    
    # Kafka configuration
    admin_client = KafkaAdminClient(
        bootstrap_servers=['localhost:9092'],
        client_id='fraudguard-setup'
    )
    
    # Define topics to create
    topics_to_create = [
        NewTopic(
            name=KAFKA_TOPICS["CDR_TOPIC"],
            num_partitions=3,
            replication_factor=1
        ),
        NewTopic(
            name=KAFKA_TOPICS["ALERTS_TOPIC"],
            num_partitions=2,
            replication_factor=1
        ),
        NewTopic(
            name=KAFKA_TOPICS["PATTERNS_TOPIC"],
            num_partitions=2,
            replication_factor=1
        ),
        NewTopic(
            name=KAFKA_TOPICS["NETWORK_UPDATES"],
            num_partitions=2,
            replication_factor=1
        )
    ]
    
    # Create topics
    for topic in topics_to_create:
        try:
            admin_client.create_topics([topic])
            logger.info(f"Created topic: {topic.name}")
        except TopicAlreadyExistsError:
            logger.info(f"Topic already exists: {topic.name}")
        except Exception as e:
            logger.error(f"Error creating topic {topic.name}: {e}")
    
    logger.info("Kafka topics setup completed")


if __name__ == "__main__":
    # Wait for Kafka to be ready
    logger.info("Waiting for Kafka to be ready...")
    time.sleep(10)
    
    create_kafka_topics()