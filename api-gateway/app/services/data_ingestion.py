"""
Data Ingestion Service for FraudGuard 360
Handles CDR data ingestion, validation, and streaming to Kafka
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import asyncio
import json
import csv
import pandas as pd
from datetime import datetime, timezone
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging
import os
import hashlib
import io
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
CDR_TOPIC = os.getenv("CDR_TOPIC", "telecom-cdr-topic")

class CallType(str, Enum):
    VOICE = "voice"
    SMS = "sms"
    DATA = "data"
    INTERNATIONAL = "international"
    PREMIUM = "premium"
    ROAMING = "roaming"

class CDRRecord(BaseModel):
    """Call Detail Record model"""
    call_id: str = Field(..., description="Unique call identifier")
    timestamp: datetime = Field(..., description="Call timestamp")
    caller_id: str = Field(..., description="Calling party identifier")
    callee_id: str = Field(..., description="Called party identifier")
    call_type: CallType = Field(..., description="Type of call")
    duration: int = Field(..., ge=0, description="Call duration in seconds")
    cost: float = Field(..., ge=0, description="Call cost")
    caller_location: str = Field(..., description="Caller location")
    callee_location: str = Field(..., description="Callee location")
    network_cell_id: str = Field(..., description="Network cell tower ID")
    device_imei: Optional[str] = Field(None, description="Device IMEI")
    roaming_flag: bool = Field(default=False, description="Roaming indicator")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v > datetime.now(timezone.utc):
            raise ValueError('Timestamp cannot be in the future')
        return v
    
    @validator('caller_id', 'callee_id')
    def validate_phone_number(cls, v):
        if not v or len(v) < 10:
            raise ValueError('Invalid phone number format')
        return v

class BatchCDRRequest(BaseModel):
    """Batch CDR ingestion request"""
    records: List[CDRRecord] = Field(..., description="List of CDR records")
    batch_id: str = Field(..., description="Batch identifier")
    source_system: str = Field(..., description="Source system identifier")

class IngestionResponse(BaseModel):
    """Ingestion response model"""
    success: bool
    message: str
    processed_count: int
    failed_count: int
    batch_id: Optional[str] = None
    errors: List[str] = []

class CDRDataIngestionService:
    """Main CDR data ingestion service"""
    
    def __init__(self):
        self.kafka_producer = None
        self.processed_count = 0
        self.failed_count = 0
        self._initialize_kafka()
    
    def _initialize_kafka(self):
        """Initialize Kafka producer"""
        try:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                batch_size=16384,
                linger_ms=10,
                buffer_memory=33554432
            )
            logger.info(f"Kafka producer initialized with servers: {KAFKA_BOOTSTRAP_SERVERS}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            self.kafka_producer = None
    
    async def ingest_single_cdr(self, cdr: CDRRecord) -> bool:
        """Ingest a single CDR record"""
        try:
            # Validate and enrich CDR
            enriched_cdr = await self._enrich_cdr(cdr)
            
            # Send to Kafka
            if self.kafka_producer:
                future = self.kafka_producer.send(
                    CDR_TOPIC,
                    key=cdr.call_id,
                    value=enriched_cdr
                )
                # Don't wait for confirmation in async context
                self.kafka_producer.flush()
            
            self.processed_count += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest CDR {cdr.call_id}: {e}")
            self.failed_count += 1
            return False
    
    async def ingest_batch_cdr(self, batch: BatchCDRRequest) -> IngestionResponse:
        """Ingest a batch of CDR records"""
        logger.info(f"Starting batch ingestion: {batch.batch_id} with {len(batch.records)} records")
        
        successful_records = 0
        failed_records = 0
        errors = []
        
        # Process records in batches for better performance
        batch_size = 100
        for i in range(0, len(batch.records), batch_size):
            batch_chunk = batch.records[i:i + batch_size]
            
            # Process chunk concurrently
            tasks = [self.ingest_single_cdr(cdr) for cdr in batch_chunk]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_records += 1
                    errors.append(f"Record {i+j}: {str(result)}")
                elif result:
                    successful_records += 1
                else:
                    failed_records += 1
                    errors.append(f"Record {i+j}: Unknown processing error")
        
        logger.info(f"Batch {batch.batch_id} completed: {successful_records} success, {failed_records} failed")
        
        return IngestionResponse(
            success=successful_records > 0,
            message=f"Processed {successful_records}/{len(batch.records)} records",
            processed_count=successful_records,
            failed_count=failed_records,
            batch_id=batch.batch_id,
            errors=errors[:10]  # Limit errors to avoid large responses
        )
    
    async def ingest_csv_file(self, file: UploadFile) -> IngestionResponse:
        """Ingest CDR data from CSV file"""
        try:
            # Read CSV file
            content = await file.read()
            csv_data = pd.read_csv(io.StringIO(content.decode('utf-8')))
            
            # Validate CSV structure
            required_columns = [
                'call_id', 'timestamp', 'caller_id', 'callee_id', 
                'call_type', 'duration', 'cost', 'caller_location', 
                'callee_location', 'network_cell_id'
            ]
            
            missing_columns = [col for col in required_columns if col not in csv_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert to CDR records
            cdr_records = []
            for _, row in csv_data.iterrows():
                try:
                    cdr = CDRRecord(
                        call_id=str(row.get('call_id')),
                        timestamp=pd.to_datetime(row.get('timestamp')),
                        caller_id=str(row.get('caller_id')),
                        callee_id=str(row.get('callee_id')),
                        call_type=row.get('call_type', 'voice'),
                        duration=int(row.get('duration', 0)),
                        cost=float(row.get('cost', 0.0)),
                        caller_location=str(row.get('caller_location', '')),
                        callee_location=str(row.get('callee_location', '')),
                        network_cell_id=str(row.get('network_cell_id', '')),
                        device_imei=row.get('device_imei'),
                        roaming_flag=bool(row.get('roaming_flag', False))
                    )
                    cdr_records.append(cdr)
                except Exception as e:
                    logger.warning(f"Failed to parse row {len(cdr_records)}: {e}")
            
            # Create batch request
            batch_id = hashlib.md5(content).hexdigest()[:16]
            batch_request = BatchCDRRequest(
                records=cdr_records,
                batch_id=batch_id,
                source_system="csv_upload"
            )
            
            # Process batch
            return await self.ingest_batch_cdr(batch_request)
            
        except Exception as e:
            logger.error(f"CSV ingestion failed: {e}")
            return IngestionResponse(
                success=False,
                message=f"CSV processing failed: {str(e)}",
                processed_count=0,
                failed_count=0,
                errors=[str(e)]
            )
    
    async def _enrich_cdr(self, cdr: CDRRecord) -> Dict[str, Any]:
        """Enrich CDR with additional computed features"""
        cdr_dict = cdr.dict()
        
        # Add computed features
        cdr_dict['ingestion_timestamp'] = datetime.now(timezone.utc).isoformat()
        cdr_dict['hour_of_day'] = cdr.timestamp.hour
        cdr_dict['day_of_week'] = cdr.timestamp.weekday()
        cdr_dict['is_weekend'] = cdr.timestamp.weekday() >= 5
        cdr_dict['is_night_call'] = cdr.timestamp.hour < 6 or cdr.timestamp.hour > 22
        
        # Fraud risk indicators
        cdr_dict['risk_indicators'] = {
            'high_cost': cdr.cost > 100,
            'long_duration': cdr.duration > 3600,  # > 1 hour
            'international': cdr.call_type == CallType.INTERNATIONAL,
            'premium': cdr.call_type == CallType.PREMIUM,
            'roaming': cdr.roaming_flag
        }
        
        return cdr_dict
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get ingestion metrics"""
        return {
            'processed_count': self.processed_count,
            'failed_count': self.failed_count,
            'success_rate': self.processed_count / (self.processed_count + self.failed_count) if self.processed_count + self.failed_count > 0 else 0,
            'kafka_connected': self.kafka_producer is not None
        }

# FastAPI Application
app = FastAPI(
    title="FraudGuard 360 Data Ingestion Service",
    description="CDR data ingestion and streaming service",
    version="1.0.0"
)

# Global service instance
ingestion_service = CDRDataIngestionService()

@app.post("/api/v1/cdr/ingest", response_model=IngestionResponse)
async def ingest_single_cdr_endpoint(cdr: CDRRecord):
    """Ingest a single CDR record"""
    success = await ingestion_service.ingest_single_cdr(cdr)
    return IngestionResponse(
        success=success,
        message="CDR record processed" if success else "CDR processing failed",
        processed_count=1 if success else 0,
        failed_count=0 if success else 1
    )

@app.post("/api/v1/cdr/batch", response_model=IngestionResponse)
async def ingest_batch_cdr_endpoint(batch: BatchCDRRequest):
    """Ingest a batch of CDR records"""
    return await ingestion_service.ingest_batch_cdr(batch)

@app.post("/api/v1/cdr/upload", response_model=IngestionResponse)
async def upload_cdr_file(file: UploadFile = File(...)):
    """Upload and ingest CDR data from CSV file"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    return await ingestion_service.ingest_csv_file(file)

@app.get("/api/v1/cdr/metrics")
async def get_ingestion_metrics():
    """Get CDR ingestion metrics"""
    return ingestion_service.get_metrics()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kafka_connected": ingestion_service.kafka_producer is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)