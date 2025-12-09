# FraudGuard-360 OCI Infrastructure

This directory contains Terraform configuration for provisioning the core network infrastructure on Oracle Cloud Infrastructure (OCI).

## Prerequisites

1. **OCI Account**: Active Oracle Cloud account with appropriate permissions
2. **Terraform**: Version 1.0 or higher installed
3. **OCI CLI**: Configured with API keys (optional but recommended)

## Infrastructure Components

This Terraform configuration provisions:

- **Virtual Cloud Network (VCN)**: `10.0.0.0/16`
- **Public Subnet**: `10.0.1.0/24` - For load balancers and public-facing services
- **Private Subnet**: `10.0.2.0/24` - For databases and backend services
- **Internet Gateway**: For public internet access
- **Route Table**: Routes traffic to the internet gateway
- **Security Lists**: 
  - SSH (port 22)
  - HTTP (port 80)
  - HTTPS (port 443)
  - Kubernetes API (port 6443)
  - NodePort range (30000-32767)

## Configuration

### 1. Set up OCI credentials

Create a `terraform.tfvars` file:

```hcl
tenancy_ocid     = "ocid1.tenancy.oc1..aaaaaaaa..."
user_ocid        = "ocid1.user.oc1..aaaaaaaa..."
fingerprint      = "xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx"
private_key_path = "~/.oci/oci_api_key.pem"
region           = "us-ashburn-1"
compartment_ocid = "ocid1.compartment.oc1..aaaaaaaa..."

# Optional: Override defaults
project_name         = "fraudguard"
environment          = "production"
vcn_cidr             = "10.0.0.0/16"
public_subnet_cidr   = "10.0.1.0/24"
private_subnet_cidr  = "10.0.2.0/24"
```

**Important**: Never commit `terraform.tfvars` to version control!

### 2. Or use environment variables

```bash
export TF_VAR_tenancy_ocid="ocid1.tenancy.oc1..aaaaaaaa..."
export TF_VAR_user_ocid="ocid1.user.oc1..aaaaaaaa..."
export TF_VAR_fingerprint="xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx"
export TF_VAR_private_key_path="~/.oci/oci_api_key.pem"
export TF_VAR_region="us-ashburn-1"
export TF_VAR_compartment_ocid="ocid1.compartment.oc1..aaaaaaaa..."
```

## Usage

### Initialize Terraform

```bash
cd terraform
terraform init
```

### Plan the deployment

```bash
terraform plan
```

### Apply the configuration

```bash
terraform apply
```

### View outputs

```bash
terraform output
```

### Destroy the infrastructure

```bash
terraform destroy
```

## Outputs

After successful deployment, Terraform will output:

- `vcn_id`: The OCID of the created VCN
- `public_subnet_id`: The OCID of the public subnet
- `private_subnet_id`: The OCID of the private subnet
- `internet_gateway_id`: The OCID of the internet gateway
- `security_list_id`: The OCID of the security list
- `infrastructure_summary`: Complete summary of deployed resources

## Network Architecture

```
┌─────────────────────────────────────────────────────┐
│          VCN: 10.0.0.0/16 (fraudguard-vcn)         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  Public Subnet: 10.0.1.0/24                 │  │
│  │  - API Gateway Load Balancer                │  │
│  │  - Kubernetes Worker Nodes                  │  │
│  │  - Bastion Host                             │  │
│  └──────────────────────────────────────────────┘  │
│                        │                            │
│                        │                            │
│  ┌──────────────────────────────────────────────┐  │
│  │  Private Subnet: 10.0.2.0/24                │  │
│  │  - PostgreSQL Database                      │  │
│  │  - Redis Cache                              │  │
│  │  - Neo4j Graph Database                     │  │
│  │  - Kafka Cluster                            │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
                 Internet Gateway
                         │
                         ▼
                     Internet
```

## Security Considerations

1. **API Keys**: Store API keys securely, never commit to version control
2. **Security Lists**: Review and adjust security list rules for your specific requirements
3. **CIDR Blocks**: Restrict `allowed_k8s_cidr` to your specific IP range in production
4. **Private Subnet**: Database and backend services are isolated in the private subnet
5. **Tags**: All resources are tagged for easy identification and cost tracking

## Next Steps

After provisioning the network infrastructure:

1. **Deploy Kubernetes Cluster**: Use OKE (Oracle Kubernetes Engine) or self-managed cluster
2. **Set up Database Services**: Deploy PostgreSQL, Redis, Neo4j, and Kafka
3. **Configure DNS**: Set up DNS records for your domain
4. **Deploy Application**: Use Helm charts from `../helm/fraudguard/`
5. **Set up Monitoring**: Deploy Prometheus and Grafana

## Cost Estimation

Estimated monthly costs for this infrastructure (USD):

- VCN: Free
- Internet Gateway: Free
- Subnets: Free
- Security Lists: Free

**Total Network Infrastructure**: ~$0/month

Additional costs will apply when you deploy compute, database, and other resources.

## Support

For issues or questions:
- Create an issue in the GitHub repository
- Consult OCI documentation: https://docs.oracle.com/en-us/iaas/
- Terraform OCI Provider docs: https://registry.terraform.io/providers/oracle/oci/latest/docs

## License

MIT License - See LICENSE file in the repository root
