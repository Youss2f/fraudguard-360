# ==============================================================================
# FraudGuard 360 - Infrastructure Deployment Guide
# Comprehensive guide for deploying the infrastructure
# ==============================================================================

# Infrastructure Deployment Guide

This guide provides step-by-step instructions for deploying the FraudGuard 360 infrastructure using Terraform.

## Prerequisites

Before deploying the infrastructure, ensure you have the following:

1. **AWS CLI** configured with appropriate credentials
2. **Terraform** (>= 1.0) installed
3. **kubectl** installed for Kubernetes management
4. **Helm** (>= 3.0) for package management
5. **Appropriate AWS permissions** for creating infrastructure resources

## AWS Permissions Required

Your AWS user/role needs the following permissions:
- EC2 (VPC, Subnets, Security Groups, Load Balancers)
- EKS (Cluster and Node Group management)
- RDS (Database instances and parameter groups)
- ElastiCache (Redis clusters)
- MSK (Kafka clusters)
- IAM (Roles and policies creation)
- KMS (Key management)
- CloudWatch (Logging and monitoring)
- Secrets Manager (Credential storage)
- Auto Scaling (Launch templates and groups)

## Directory Structure

```
infrastructure/
├── main.tf                    # Root module configuration
├── variables.tf               # Root module variables
├── outputs.tf                 # Root module outputs
├── modules/                   # Terraform modules
│   ├── k8s/                  # EKS cluster module
│   ├── kafka/                # MSK Kafka module
│   ├── neo4j/                # Neo4j graph database module
│   ├── rds/                  # PostgreSQL database module
│   └── monitoring/           # Monitoring stack module
└── environments/
    └── dev/                  # Development environment
        ├── main.tf
        ├── variables.tf
        ├── outputs.tf
        └── terraform.tfvars
```

## Deployment Steps

### Step 1: Prepare the Environment

1. **Set up Terraform Backend** (Optional but Recommended)
   ```bash
   # Create S3 bucket for state storage
   aws s3 mb s3://fraudguard-360-terraform-state-dev --region us-east-1
   
   # Enable versioning
   aws s3api put-bucket-versioning \
     --bucket fraudguard-360-terraform-state-dev \
     --versioning-configuration Status=Enabled
   
   # Create DynamoDB table for state locking
   aws dynamodb create-table \
     --table-name fraudguard-360-terraform-locks \
     --attribute-definitions AttributeName=LockID,AttributeType=S \
     --key-schema AttributeName=LockID,KeyType=HASH \
     --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
     --region us-east-1
   ```

2. **Configure AWS CLI**
   ```bash
   aws configure
   # Enter your AWS Access Key ID, Secret Access Key, and region
   ```

### Step 2: Deploy Development Environment

1. **Navigate to development environment**
   ```bash
   cd infrastructure/environments/dev
   ```

2. **Initialize Terraform**
   ```bash
   terraform init
   ```

3. **Review the deployment plan**
   ```bash
   terraform plan -var-file=terraform.tfvars
   ```

4. **Deploy the infrastructure**
   ```bash
   terraform apply -var-file=terraform.tfvars
   ```
   
   Type `yes` when prompted to confirm the deployment.

### Step 3: Post-Deployment Configuration

1. **Configure kubectl**
   ```bash
   aws eks update-kubeconfig --region us-east-1 --name dev-fraudguard-cluster
   ```

2. **Verify cluster connectivity**
   ```bash
   kubectl get nodes
   kubectl get namespaces
   ```

3. **Check monitoring stack**
   ```bash
   kubectl get pods -n monitoring
   kubectl get services -n monitoring
   ```

4. **Access Grafana Dashboard**
   ```bash
   # If using LoadBalancer service type
   kubectl get svc -n monitoring grafana
   # Note the external IP/hostname
   
   # Or use port forwarding
   kubectl port-forward -n monitoring svc/grafana 3000:3000
   # Access at http://localhost:3000
   # Default credentials: admin/fraudguard-dev-2024
   ```

## Service Endpoints

After successful deployment, you'll have access to:

| Service | Port | Access Method |
|---------|------|---------------|
| Grafana | 3000 | LoadBalancer or Port Forward |
| Prometheus | 9090 | Port Forward |
| AlertManager | 9093 | Port Forward |
| Neo4j HTTP | 7474 | Load Balancer DNS |
| Neo4j HTTPS | 7473 | Load Balancer DNS |
| Neo4j Bolt | 7687 | Load Balancer DNS |

## Environment-Specific Configurations

### Development Environment
- **Resource Sizing**: Smaller instances for cost optimization
- **Backup Retention**: 3 days
- **Deletion Protection**: Disabled for easy teardown
- **Monitoring**: Basic monitoring with cost-optimized settings
- **Network**: Single NAT Gateway for cost savings

### Production Environment (Future)
- **Resource Sizing**: Production-grade instances
- **Backup Retention**: 30 days
- **Deletion Protection**: Enabled
- **Monitoring**: Comprehensive monitoring and alerting
- **Network**: Multi-AZ NAT Gateways for high availability

## Customization

### Modifying Resource Sizes
Edit `infrastructure/environments/dev/terraform.tfvars`:

```hcl
# Example: Increase EKS node capacity
node_groups = {
  general = {
    name           = "general"
    instance_types = ["t3.large"]    # Changed from t3.medium
    desired_size   = 3               # Changed from 2
    max_size       = 10              # Changed from 5
    min_size       = 2               # Changed from 1
  }
}

# Example: Increase RDS instance size
rds_instance_class = "db.t3.small"    # Changed from db.t3.micro
```

### Adding Custom Monitoring
1. Edit `infrastructure/modules/monitoring/main.tf`
2. Add custom Grafana dashboards
3. Configure additional alert rules
4. Redeploy: `terraform apply`

## Troubleshooting

### Common Issues

1. **Insufficient Permissions**
   ```
   Error: AccessDenied: User: arn:aws:iam::xxx is not authorized to perform: eks:CreateCluster
   ```
   **Solution**: Ensure your AWS user has the required permissions listed above.

2. **Resource Limits**
   ```
   Error: LimitExceeded: You have reached the limit on the number of VPCs
   ```
   **Solution**: Check AWS service limits and request increases if needed.

3. **State Lock Issues**
   ```
   Error: Error locking state: ConditionalCheckFailedException
   ```
   **Solution**: 
   ```bash
   terraform force-unlock <LOCK_ID>
   ```

4. **EKS Node Group Creation Fails**
   ```
   Error: NodeCreationFailure: Instances failed to join the kubernetes cluster
   ```
   **Solution**: Check VPC configuration and ensure private subnets have internet access through NAT Gateway.

### Validation Commands

```bash
# Check infrastructure status
terraform show

# Validate Terraform configuration
terraform validate

# Check Kubernetes cluster
kubectl cluster-info
kubectl get nodes --show-labels

# Test database connectivity
kubectl run psql-test --image=postgres:15 --rm -it --restart=Never -- \
  psql -h <RDS_ENDPOINT> -U postgres -d fraudguard

# Test Redis connectivity
kubectl run redis-test --image=redis:7-alpine --rm -it --restart=Never -- \
  redis-cli -h <REDIS_ENDPOINT> ping
```

## Cleanup

To destroy the infrastructure:

```bash
cd infrastructure/environments/dev
terraform destroy -var-file=terraform.tfvars
```

**Warning**: This will permanently delete all resources. Ensure you have backups if needed.

## Next Steps

1. **Deploy Applications**: Use the Kubernetes manifests in the `/k8s` directory
2. **Configure CI/CD**: Set up GitHub Actions for automated deployments
3. **Set up Monitoring**: Configure custom dashboards and alerts
4. **Security Hardening**: Implement additional security measures for production
5. **Backup Strategy**: Configure automated backups for critical data

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Terraform and AWS documentation
3. Check CloudWatch logs for detailed error messages
4. Use `terraform show` and `kubectl describe` for resource details

## Security Considerations

1. **Secrets Management**: Database passwords are stored in AWS Secrets Manager
2. **Network Security**: Private subnets for compute resources
3. **Encryption**: At-rest and in-transit encryption enabled
4. **IAM Roles**: Least privilege access principles
5. **Security Groups**: Restrictive inbound rules

Remember to regularly update and patch your infrastructure for security and performance improvements.