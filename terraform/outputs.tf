# Output Values for FraudGuard-360 Infrastructure

output "vcn_id" {
  description = "The OCID of the VCN"
  value       = oci_core_vcn.fraudguard_vcn.id
}

output "vcn_cidr" {
  description = "The CIDR block of the VCN"
  value       = oci_core_vcn.fraudguard_vcn.cidr_blocks[0]
}

output "public_subnet_id" {
  description = "The OCID of the public subnet"
  value       = oci_core_subnet.fraudguard_public_subnet.id
}

output "public_subnet_cidr" {
  description = "The CIDR block of the public subnet"
  value       = oci_core_subnet.fraudguard_public_subnet.cidr_block
}

output "private_subnet_id" {
  description = "The OCID of the private subnet"
  value       = oci_core_subnet.fraudguard_private_subnet.id
}

output "private_subnet_cidr" {
  description = "The CIDR block of the private subnet"
  value       = oci_core_subnet.fraudguard_private_subnet.cidr_block
}

output "internet_gateway_id" {
  description = "The OCID of the Internet Gateway"
  value       = oci_core_internet_gateway.fraudguard_igw.id
}

output "security_list_id" {
  description = "The OCID of the security list"
  value       = oci_core_security_list.fraudguard_public_sl.id
}

output "route_table_id" {
  description = "The OCID of the route table"
  value       = oci_core_route_table.fraudguard_public_rt.id
}

# Summary output
output "infrastructure_summary" {
  description = "Summary of deployed infrastructure"
  value = {
    vcn_id              = oci_core_vcn.fraudguard_vcn.id
    vcn_cidr            = oci_core_vcn.fraudguard_vcn.cidr_blocks[0]
    public_subnet_id    = oci_core_subnet.fraudguard_public_subnet.id
    private_subnet_id   = oci_core_subnet.fraudguard_private_subnet.id
    region              = var.region
    environment         = var.environment
  }
}
