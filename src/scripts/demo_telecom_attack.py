#!/usr/bin/env python3
"""
FraudGuard-360 Telecom Attack Simulation
=========================================

This script simulates realistic telecom fraud patterns for demonstration purposes.
It generates CDRs for two common fraud scenarios:

1. WANGIRI ATTACK: Single international number bombarding local subscribers
   with missed calls to trigger expensive callbacks.

2. SIM BOX FRAUD: Multiple SIM cards sharing the same IMEI device,
   routing international calls through illegal GSM gateways.

Usage:
    python demo_telecom_attack.py [--api-url URL] [--dry-run]

Author: FraudGuard-360 Team
"""

import argparse
import asyncio
import random
import string
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

try:
    import httpx
except ImportError:
    print("[ERROR] httpx not installed. Run: pip install httpx")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SimulationConfig:
    """Configuration for the attack simulation."""
    api_url: str = "http://localhost:8000"
    
    # Victim Pool
    victim_count: int = 50
    victim_prefix: str = "+212"  # Morocco
    
    # Wangiri Attack Parameters
    wangiri_attacker_prefix: str = "+220"  # Gambia (common Wangiri source)
    wangiri_call_count: int = 200
    wangiri_duration_range: tuple = (0, 1)  # Missed calls
    wangiri_interval_ms: int = 50  # High CPS (20 calls/sec)
    
    # SIM Box Parameters
    simbox_msisdn_count: int = 10
    simbox_prefix: str = "+212"  # Local numbers
    simbox_call_count: int = 100
    simbox_shared_imei: str = "353456789012345"  # Single device
    simbox_cell_tower: str = "CELL_CASABLANCA_ILLEGAL_001"
    simbox_destinations: List[str] = field(default_factory=lambda: ["+33", "+44", "+49", "+1"])
    simbox_duration_range: tuple = (120, 600)  # Long calls
    
    # Timing
    dry_run: bool = False
    verbose: bool = True


# =============================================================================
# DATA GENERATORS
# =============================================================================

def generate_msisdn(prefix: str, length: int = 9) -> str:
    """Generate a random phone number with given prefix."""
    suffix = ''.join(random.choices(string.digits, k=length))
    return f"{prefix}{suffix}"


def generate_imei() -> str:
    """Generate a random valid IMEI number."""
    # TAC (8 digits) + Serial (6 digits) + Check digit
    tac = ''.join(random.choices(string.digits, k=8))
    serial = ''.join(random.choices(string.digits, k=6))
    return f"{tac}{serial}0"


def generate_cell_tower_id(region: str = "CASABLANCA") -> str:
    """Generate a random cell tower ID."""
    sector = random.choice(['A', 'B', 'C'])
    tower_num = random.randint(1, 999)
    return f"CELL_{region}_{sector}{tower_num:03d}"


# =============================================================================
# CDR MODELS
# =============================================================================

@dataclass
class CDR:
    """Call Detail Record for API submission."""
    caller_msisdn: str
    callee_msisdn: str
    duration: int
    call_type: str = "voice"
    cell_tower_id: str = ""
    imei: str = ""
    
    def to_dict(self) -> dict:
        """Convert to API-compatible dictionary."""
        return {
            "caller_msisdn": self.caller_msisdn,
            "callee_msisdn": self.callee_msisdn,
            "duration": self.duration,
            "call_type": self.call_type,
            "cell_tower_id": self.cell_tower_id or generate_cell_tower_id(),
            "imei": self.imei or generate_imei(),
        }


# =============================================================================
# ATTACK SIMULATORS
# =============================================================================

class WangiriAttackSimulator:
    """
    Simulates a Wangiri (one-ring) attack pattern.
    
    Characteristics:
    - Single source number (international premium rate)
    - Mass calls to local victims
    - Very short duration (0-1 second missed calls)
    - High call frequency (CPS)
    """
    
    def __init__(self, config: SimulationConfig, victims: List[str]):
        self.config = config
        self.victims = victims
        self.attacker_msisdn = generate_msisdn(config.wangiri_attacker_prefix)
        self.cdrs: List[CDR] = []
        
    def generate_cdrs(self) -> List[CDR]:
        """Generate Wangiri attack CDRs."""
        print(f"\n[INFO] Generating Wangiri Attack CDRs...")
        print(f"       Attacker: {self.attacker_msisdn}")
        print(f"       Target Pool: {len(self.victims)} victims")
        print(f"       Call Count: {self.config.wangiri_call_count}")
        
        for i in range(self.config.wangiri_call_count):
            victim = random.choice(self.victims)
            duration = random.randint(*self.config.wangiri_duration_range)
            
            cdr = CDR(
                caller_msisdn=self.attacker_msisdn,
                callee_msisdn=victim,
                duration=duration,
                call_type="voice",
                cell_tower_id=generate_cell_tower_id("INTERNATIONAL"),
                imei=generate_imei(),  # Different devices (spoofed)
            )
            self.cdrs.append(cdr)
            
        print(f"[SUCCESS] Generated {len(self.cdrs)} Wangiri CDRs")
        return self.cdrs


class SIMBoxSimulator:
    """
    Simulates SIM Box fraud pattern.
    
    Characteristics:
    - Multiple SIM cards (different MSISDNs)
    - Single shared device (same IMEI)
    - Static location (single cell tower)
    - International destinations (bypass termination fees)
    - Longer call durations
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.sim_cards: List[str] = []
        self.cdrs: List[CDR] = []
        self._generate_sim_cards()
        
    def _generate_sim_cards(self):
        """Generate the pool of SIM cards in the box."""
        for _ in range(self.config.simbox_msisdn_count):
            msisdn = generate_msisdn(self.config.simbox_prefix)
            self.sim_cards.append(msisdn)
            
    def generate_cdrs(self) -> List[CDR]:
        """Generate SIM Box fraud CDRs."""
        print(f"\n[INFO] Generating SIM Box Fraud CDRs...")
        print(f"       SIM Cards: {len(self.sim_cards)}")
        print(f"       Shared IMEI: {self.config.simbox_shared_imei}")
        print(f"       Cell Tower: {self.config.simbox_cell_tower}")
        print(f"       Call Count: {self.config.simbox_call_count}")
        
        for i in range(self.config.simbox_call_count):
            # Rotate through SIM cards
            caller = self.sim_cards[i % len(self.sim_cards)]
            
            # International destination
            dest_prefix = random.choice(self.config.simbox_destinations)
            callee = generate_msisdn(dest_prefix)
            
            duration = random.randint(*self.config.simbox_duration_range)
            
            cdr = CDR(
                caller_msisdn=caller,
                callee_msisdn=callee,
                duration=duration,
                call_type="voice",
                cell_tower_id=self.config.simbox_cell_tower,  # STATIC!
                imei=self.config.simbox_shared_imei,  # SHARED!
            )
            self.cdrs.append(cdr)
            
        print(f"[SUCCESS] Generated {len(self.cdrs)} SIM Box CDRs")
        return self.cdrs


# =============================================================================
# API CLIENT
# =============================================================================

class FraudGuardClient:
    """Async client for FraudGuard-360 API."""
    
    def __init__(self, base_url: str, verbose: bool = True):
        self.base_url = base_url.rstrip('/')
        self.verbose = verbose
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "latencies": [],
        }
        
    async def health_check(self) -> bool:
        """Check if API is available."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    data = response.json()
                    print(f"[INFO] API Health: {data.get('status', 'unknown')}")
                    return True
                return False
        except Exception as e:
            print(f"[ERROR] Health check failed: {e}")
            return False
            
    async def submit_cdr(self, client: httpx.AsyncClient, cdr: CDR) -> bool:
        """Submit a single CDR to the API."""
        start = time.time()
        try:
            response = await client.post(
                f"{self.base_url}/v1/cdrs",
                json=cdr.to_dict(),
                timeout=30.0,
            )
            latency = (time.time() - start) * 1000
            self.stats["latencies"].append(latency)
            self.stats["total"] += 1
            
            if response.status_code in (200, 201):
                self.stats["success"] += 1
                return True
            else:
                self.stats["failed"] += 1
                if self.verbose:
                    print(f"[WARN] CDR rejected: {response.status_code} - {response.text[:100]}")
                return False
                
        except Exception as e:
            self.stats["total"] += 1
            self.stats["failed"] += 1
            if self.verbose:
                print(f"[ERROR] Request failed: {e}")
            return False
            
    async def submit_batch(self, cdrs: List[CDR], batch_name: str, interval_ms: int = 100):
        """Submit a batch of CDRs with controlled rate."""
        print(f"\n[INFO] Submitting {batch_name}...")
        print(f"       CDRs: {len(cdrs)}")
        print(f"       Interval: {interval_ms}ms")
        
        async with httpx.AsyncClient() as client:
            for i, cdr in enumerate(cdrs):
                await self.submit_cdr(client, cdr)
                
                # Progress indicator
                if (i + 1) % 50 == 0:
                    print(f"       Progress: {i + 1}/{len(cdrs)}")
                    
                # Rate limiting
                if interval_ms > 0:
                    await asyncio.sleep(interval_ms / 1000)
                    
        print(f"[SUCCESS] {batch_name} complete")
        
    def print_stats(self):
        """Print submission statistics."""
        print("\n" + "=" * 60)
        print("SUBMISSION STATISTICS")
        print("=" * 60)
        print(f"  Total CDRs:     {self.stats['total']}")
        print(f"  Successful:     {self.stats['success']}")
        print(f"  Failed:         {self.stats['failed']}")
        
        if self.stats["latencies"]:
            avg_latency = sum(self.stats["latencies"]) / len(self.stats["latencies"])
            min_latency = min(self.stats["latencies"])
            max_latency = max(self.stats["latencies"])
            print(f"  Avg Latency:    {avg_latency:.2f}ms")
            print(f"  Min Latency:    {min_latency:.2f}ms")
            print(f"  Max Latency:    {max_latency:.2f}ms")
        print("=" * 60)


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

async def run_simulation(config: SimulationConfig):
    """Run the complete attack simulation."""
    
    print("\n" + "=" * 60)
    print("FRAUDGUARD-360 TELECOM ATTACK SIMULATION")
    print("=" * 60)
    print(f"  API URL:        {config.api_url}")
    print(f"  Dry Run:        {config.dry_run}")
    print(f"  Timestamp:      {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Initialize API client
    client = FraudGuardClient(config.api_url, verbose=config.verbose)
    
    # Health check
    if not config.dry_run:
        print("\n[INFO] Performing API health check...")
        if not await client.health_check():
            print("[ERROR] API is not available. Use --dry-run to generate CDRs without submission.")
            return
            
    # Step 1: Generate victim pool
    print("\n[INFO] Generating victim pool...")
    victims = [generate_msisdn(config.victim_prefix) for _ in range(config.victim_count)]
    print(f"[SUCCESS] Generated {len(victims)} local subscribers")
    
    # Step 2: Wangiri Attack
    wangiri = WangiriAttackSimulator(config, victims)
    wangiri_cdrs = wangiri.generate_cdrs()
    
    # Step 3: SIM Box Fraud
    simbox = SIMBoxSimulator(config)
    simbox_cdrs = simbox.generate_cdrs()
    
    # Step 4: Submit to API
    if config.dry_run:
        print("\n[INFO] DRY RUN - Skipping API submission")
        print(f"       Would submit {len(wangiri_cdrs)} Wangiri CDRs")
        print(f"       Would submit {len(simbox_cdrs)} SIM Box CDRs")
        
        # Print sample CDRs
        print("\n[INFO] Sample Wangiri CDR:")
        print(f"       {wangiri_cdrs[0].to_dict()}")
        print("\n[INFO] Sample SIM Box CDR:")
        print(f"       {simbox_cdrs[0].to_dict()}")
    else:
        # Submit Wangiri attack (high frequency)
        await client.submit_batch(
            wangiri_cdrs,
            "Wangiri Attack",
            interval_ms=config.wangiri_interval_ms,
        )
        
        # Submit SIM Box fraud (normal rate)
        await client.submit_batch(
            simbox_cdrs,
            "SIM Box Fraud",
            interval_ms=100,
        )
        
        # Print statistics
        client.print_stats()
        
    # Summary
    print("\n" + "=" * 60)
    print("ATTACK SIMULATION COMPLETE")
    print("=" * 60)
    print(f"  Wangiri Attacker:  {wangiri.attacker_msisdn}")
    print(f"  SIM Box IMEI:      {config.simbox_shared_imei}")
    print(f"  SIM Box Tower:     {config.simbox_cell_tower}")
    print("=" * 60)
    print("\n[INFO] Check FSOC Dashboard at http://localhost:3001 for alerts")
    print("[INFO] Query Neo4j for graph patterns")


def main():
    """Entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="FraudGuard-360 Telecom Attack Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_telecom_attack.py                      # Run with defaults
  python demo_telecom_attack.py --dry-run            # Generate CDRs without submission
  python demo_telecom_attack.py --api-url http://api:8000  # Custom API URL
        """
    )
    
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="FraudGuard API URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate CDRs without submitting to API"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--wangiri-count",
        type=int,
        default=200,
        help="Number of Wangiri calls (default: 200)"
    )
    parser.add_argument(
        "--simbox-count",
        type=int,
        default=100,
        help="Number of SIM Box calls (default: 100)"
    )
    
    args = parser.parse_args()
    
    config = SimulationConfig(
        api_url=args.api_url,
        dry_run=args.dry_run,
        verbose=not args.quiet,
        wangiri_call_count=args.wangiri_count,
        simbox_call_count=args.simbox_count,
    )
    
    try:
        asyncio.run(run_simulation(config))
    except KeyboardInterrupt:
        print("\n[INFO] Simulation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
