# FraudGuard 360°

## Developer Onboarding

1. Clone repo: `git clone <repo>`
2. Install deps: In frontend: `npm i`; In Python dirs: `pip install -r requirements.txt`; For Flink: `mvn package`
3. Run `docker-compose up`
4. Access: Frontend at localhost:3000, API at 8000, Neo4j browser at 7474 (user/password: neo4j/password)
5. Train ML: Run `python ml-service/training/train.py` (after seeding Neo4j)
6. Submit Flink job: Use Flink UI at 8081 or `flink run -m localhost:6123 flink-jobs/target/flink-jobs-1.0.0.jar`
7. Seed data: Use Cypher in Neo4j browser or script: `LOAD CSV FROM 'file:///cdrs.csv' AS row MERGE (a:Subscriber {id: row.caller_id}) MERGE (b:Subscriber {id: row.callee_id}) MERGE (a)-[:CALL {duration: row.duration}]->(b)`
8. IDE: VS Code with Docker, Kubernetes, Prettier, Black extensions.
9. Code Style: ESLint/Prettier for JS, Black for Python.
10. Git: Feature branches, PRs with reviews.

## Testing

- Unit: `npm test` (frontend), `pytest` (Python)
- Integration: Use docker-compose, test service interactions.
- E2E: Cypress scripts in tests/ (e.g., login, visualize graph)
- Performance: k6 load tests (scripts in tests/)
- Coverage Target: 85%

## Monitoring

Deploy Prometheus/Grafana via Helm. Dashboards for latency, throughput, etc., as per PRD.

## Deployment

- Local: docker-compose
- K8s: `helm install fraudguard helm-chart/ --namespace fraudguard`
- Terraform: `terraform init && terraform apply`
