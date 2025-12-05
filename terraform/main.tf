# FraudGuard-360 Infrastructure on Oracle Cloud Infrastructure (OCI)
# This Terraform configuration provisions the core network infrastructure
# Version constraints are defined in versions.tf

# Configure the OCI Provider
provider "oci" {
  # Authentication is configured via environment variables:
  # - TF_VAR_tenancy_ocid
  # - TF_VAR_user_ocid
  # - TF_VAR_fingerprint
  # - TF_VAR_private_key_path
  # - TF_VAR_region
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}

# Create a Virtual Cloud Network (VCN)
resource "oci_core_vcn" "fraudguard_vcn" {
  compartment_id = var.compartment_ocid
  display_name   = "${var.project_name}-vcn"
  cidr_blocks    = [var.vcn_cidr]
  dns_label      = "fraudguard"

  tags = merge(
    var.common_tags,
    {
      Name = "${var.project_name}-vcn"
    }
  )
}

# Create an Internet Gateway
resource "oci_core_internet_gateway" "fraudguard_igw" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.fraudguard_vcn.id
  display_name   = "${var.project_name}-igw"
  enabled        = true

  tags = merge(
    var.common_tags,
    {
      Name = "${var.project_name}-internet-gateway"
    }
  )
}

# Create a Route Table
resource "oci_core_route_table" "fraudguard_public_rt" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.fraudguard_vcn.id
  display_name   = "${var.project_name}-public-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.fraudguard_igw.id
  }

  tags = merge(
    var.common_tags,
    {
      Name = "${var.project_name}-public-route-table"
    }
  )
}

# Create a Security List
resource "oci_core_security_list" "fraudguard_public_sl" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.fraudguard_vcn.id
  display_name   = "${var.project_name}-public-sl"

  # Egress Rules - Allow all outbound traffic
  egress_security_rules {
    destination      = "0.0.0.0/0"
    protocol         = "all"
    stateless        = false
    destination_type = "CIDR_BLOCK"
  }

  # Ingress Rules
  # Allow SSH (port 22)
  ingress_security_rules {
    protocol    = "6" # TCP
    source      = "0.0.0.0/0"
    stateless   = false
    source_type = "CIDR_BLOCK"

    tcp_options {
      min = 22
      max = 22
    }
  }

  # Allow HTTP (port 80)
  ingress_security_rules {
    protocol    = "6" # TCP
    source      = "0.0.0.0/0"
    stateless   = false
    source_type = "CIDR_BLOCK"

    tcp_options {
      min = 80
      max = 80
    }
  }

  # Allow HTTPS (port 443)
  ingress_security_rules {
    protocol    = "6" # TCP
    source      = "0.0.0.0/0"
    stateless   = false
    source_type = "CIDR_BLOCK"

    tcp_options {
      min = 443
      max = 443
    }
  }

  # Allow Kubernetes API Server (port 6443)
  ingress_security_rules {
    protocol    = "6" # TCP
    source      = var.allowed_k8s_cidr
    stateless   = false
    source_type = "CIDR_BLOCK"

    tcp_options {
      min = 6443
      max = 6443
    }
  }

  # Allow NodePort range (30000-32767) for Kubernetes services
  ingress_security_rules {
    protocol    = "6" # TCP
    source      = "0.0.0.0/0"
    stateless   = false
    source_type = "CIDR_BLOCK"

    tcp_options {
      min = 30000
      max = 32767
    }
  }

  tags = merge(
    var.common_tags,
    {
      Name = "${var.project_name}-public-security-list"
    }
  )
}

# Create a Public Subnet
resource "oci_core_subnet" "fraudguard_public_subnet" {
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.fraudguard_vcn.id
  cidr_block        = var.public_subnet_cidr
  display_name      = "${var.project_name}-public-subnet"
  dns_label         = "public"
  route_table_id    = oci_core_route_table.fraudguard_public_rt.id
  security_list_ids = [oci_core_security_list.fraudguard_public_sl.id]

  # Make this a public subnet
  prohibit_public_ip_on_vnic = false

  tags = merge(
    var.common_tags,
    {
      Name = "${var.project_name}-public-subnet"
      Tier = "Public"
    }
  )
}

# Create a Private Subnet for databases and backend services
resource "oci_core_subnet" "fraudguard_private_subnet" {
  compartment_id             = var.compartment_ocid
  vcn_id                     = oci_core_vcn.fraudguard_vcn.id
  cidr_block                 = var.private_subnet_cidr
  display_name               = "${var.project_name}-private-subnet"
  dns_label                  = "private"
  route_table_id             = oci_core_route_table.fraudguard_public_rt.id
  security_list_ids          = [oci_core_security_list.fraudguard_public_sl.id]
  prohibit_public_ip_on_vnic = true

  tags = merge(
    var.common_tags,
    {
      Name = "${var.project_name}-private-subnet"
      Tier = "Private"
    }
  )
}
