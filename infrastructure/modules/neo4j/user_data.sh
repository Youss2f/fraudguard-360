#!/bin/bash

# ==============================================================================
# FraudGuard 360 - Neo4j Installation Script
# Automated setup of Neo4j database with optimized configuration
# ==============================================================================

set -e

# Update system packages
apt-get update
apt-get upgrade -y

# Install required packages
apt-get install -y \
    wget \
    curl \
    gnupg \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    openjdk-11-jdk \
    htop \
    iotop \
    nvme-cli \
    awscli

# Set JAVA_HOME
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> /etc/environment
source /etc/environment

# ==============================================================================
# Configure Data Volume
# ==============================================================================

# Wait for the data volume to be attached
while [ ! -e /dev/nvme1n1 ]; do
    echo "Waiting for data volume..."
    sleep 5
done

# Format and mount the data volume
if ! blkid /dev/nvme1n1; then
    mkfs.ext4 /dev/nvme1n1
fi

# Create Neo4j data directory
mkdir -p /var/lib/neo4j
mount /dev/nvme1n1 /var/lib/neo4j

# Add to fstab for persistent mounting
echo '/dev/nvme1n1 /var/lib/neo4j ext4 defaults,nofail 0 2' >> /etc/fstab

# Set ownership
chown -R neo4j:neo4j /var/lib/neo4j

# ==============================================================================
# Install Neo4j
# ==============================================================================

# Add Neo4j repository
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | tee /etc/apt/sources.list.d/neo4j.list

# Update package list and install Neo4j
apt-get update
apt-get install -y neo4j=1:5.13.0

# Hold Neo4j package to prevent automatic updates
apt-mark hold neo4j

# ==============================================================================
# Configure Neo4j
# ==============================================================================

# Backup original configuration
cp /etc/neo4j/neo4j.conf /etc/neo4j/neo4j.conf.backup

# Create optimized Neo4j configuration
cat > /etc/neo4j/neo4j.conf << EOF
# ==============================================================================
# FraudGuard 360 - Neo4j Configuration
# Optimized for fraud detection graph workloads
# ==============================================================================

# Network configuration
server.default_listen_address=0.0.0.0
server.bolt.listen_address=0.0.0.0:7687
server.http.listen_address=0.0.0.0:7474
server.https.listen_address=0.0.0.0:7473

# Enable HTTP and Bolt connectors
server.bolt.enabled=true
server.http.enabled=true
server.https.enabled=false

# Authentication
dbms.security.auth_enabled=true

# Memory configuration
server.memory.heap.initial_size=${heap_size}
server.memory.heap.max_size=${heap_size}
server.memory.pagecache.size=${pagecache_size}

# Database location
server.directories.data=/var/lib/neo4j/data
server.directories.logs=/var/lib/neo4j/logs
server.directories.import=/var/lib/neo4j/import
server.directories.plugins=/var/lib/neo4j/plugins

# Transaction log configuration
server.tx_log.rotation.retention_policy=100M size
server.tx_log.rotation.size=25M

# Query configuration for fraud detection workloads
dbms.query.timeout=60s
dbms.transaction.timeout=60s
dbms.locks.timeout=30s

# Performance tuning
dbms.query.cache_size=1000
dbms.query.plan_cache_size=1000
dbms.db.query_cache_size=256

# Logging configuration
server.logs.debug.level=INFO
server.logs.gc.enabled=true
dbms.logs.query.enabled=true
dbms.logs.query.threshold=1s

# Security
server.http.log.enabled=true
server.security.procedures.unrestricted=apoc.*,gds.*

# Metrics and monitoring
server.metrics.enabled=true
server.metrics.jmx.enabled=true
server.metrics.prometheus.enabled=true
server.metrics.prometheus.endpoint=0.0.0.0:2004

# APOC configuration
dbms.security.procedures.allowlist=apoc.*,gds.*
dbms.security.procedures.whitelist=apoc.*,gds.*
apoc.export.file.enabled=true
apoc.import.file.enabled=true
apoc.import.file.use_neo4j_config=true

# Graph Data Science configuration
gds.enterprise.license_file=/var/lib/neo4j/gds.license
EOF

# ==============================================================================
# Install APOC and Graph Data Science
# ==============================================================================

# Download and install APOC
wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/5.13.0/apoc-5.13.0-core.jar \
     -O /var/lib/neo4j/plugins/apoc-5.13.0-core.jar

# Download and install Graph Data Science
wget https://github.com/neo4j/graph-data-science/releases/download/2.5.3/neo4j-graph-data-science-2.5.3.jar \
     -O /var/lib/neo4j/plugins/neo4j-graph-data-science-2.5.3.jar

# Set proper ownership
chown -R neo4j:neo4j /var/lib/neo4j

# ==============================================================================
# Set Initial Password
# ==============================================================================

# Start Neo4j temporarily to set password
systemctl start neo4j
sleep 30

# Set the initial password
neo4j-admin dbms set-initial-password "${neo4j_password}"

# Stop Neo4j to complete configuration
systemctl stop neo4j

# ==============================================================================
# System Optimization
# ==============================================================================

# Optimize system settings for Neo4j
cat >> /etc/sysctl.conf << EOF
# Neo4j optimizations
vm.swappiness=1
vm.dirty_ratio=40
vm.dirty_background_ratio=10
fs.file-max=1000000
EOF

# Apply sysctl settings
sysctl -p

# Set ulimits for neo4j user
cat >> /etc/security/limits.conf << EOF
neo4j   soft    nofile  60000
neo4j   hard    nofile  60000
neo4j   soft    nproc   60000
neo4j   hard    nproc   60000
EOF

# ==============================================================================
# CloudWatch Agent Setup
# ==============================================================================

# Download and install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i amazon-cloudwatch-agent.deb

# Create CloudWatch configuration
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
{
    "agent": {
        "metrics_collection_interval": 60
    },
    "metrics": {
        "namespace": "FraudGuard/Neo4j",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "/var/lib/neo4j"
                ]
            },
            "diskio": {
                "measurement": [
                    "io_time"
                ],
                "metrics_collection_interval": 60
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            }
        }
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/lib/neo4j/logs/neo4j.log",
                        "log_group_name": "/aws/ec2/neo4j/application",
                        "log_stream_name": "{instance_id}/neo4j.log"
                    },
                    {
                        "file_path": "/var/lib/neo4j/logs/debug.log",
                        "log_group_name": "/aws/ec2/neo4j/debug",
                        "log_stream_name": "{instance_id}/debug.log"
                    }
                ]
            }
        }
    }
}
EOF

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
    -s

# ==============================================================================
# Enable and Start Neo4j
# ==============================================================================

# Enable Neo4j to start on boot
systemctl enable neo4j

# Start Neo4j
systemctl start neo4j

# Wait for Neo4j to be ready
sleep 60

# Verify Neo4j is running
systemctl status neo4j

# ==============================================================================
# Health Check Script
# ==============================================================================

cat > /usr/local/bin/neo4j-health-check.sh << 'EOF'
#!/bin/bash
# Neo4j health check script

NEO4J_URI="http://localhost:7474"
STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" $NEO4J_URI/db/neo4j/tx/commit)

if [ $STATUS_CODE -eq 200 ]; then
    echo "Neo4j is healthy"
    exit 0
else
    echo "Neo4j is unhealthy (HTTP $STATUS_CODE)"
    exit 1
fi
EOF

chmod +x /usr/local/bin/neo4j-health-check.sh

# ==============================================================================
# Backup Script
# ==============================================================================

cat > /usr/local/bin/neo4j-backup.sh << 'EOF'
#!/bin/bash
# Neo4j backup script

BACKUP_DIR="/var/lib/neo4j/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="neo4j_backup_$DATE"

mkdir -p $BACKUP_DIR

# Create backup
neo4j-admin database backup --database=neo4j --to-path=$BACKUP_DIR/$BACKUP_NAME

# Compress backup
tar -czf $BACKUP_DIR/$BACKUP_NAME.tar.gz -C $BACKUP_DIR $BACKUP_NAME

# Remove uncompressed backup
rm -rf $BACKUP_DIR/$BACKUP_NAME

# Keep only last 7 backups
find $BACKUP_DIR -name "neo4j_backup_*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
EOF

chmod +x /usr/local/bin/neo4j-backup.sh

# Schedule daily backups
echo "0 2 * * * /usr/local/bin/neo4j-backup.sh" | crontab -u neo4j -

echo "Neo4j installation and configuration completed successfully!"