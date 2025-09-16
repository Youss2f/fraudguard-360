import random
import json

def generate_cdr(num=100):
    cdrs = []
    for _ in range(num):
        cdr = {
            "caller_id": str(random.randint(100000, 999999)),
            "callee_id": str(random.randint(100000, 999999)),
            "duration": random.randint(10, 300),
            "timestamp": f"2025-09-{random.randint(1,30)}T{random.randint(0,23)}:00:00"
        }
        cdrs.append(cdr)
    return cdrs

# Example usage: print(json.dumps(generate_cdr()))
