import torch
from torch_geometric.loader import NeighborLoader
from torch.optim import Adam
from neo4j import GraphDatabase
from models.graphsage import GraphSAGE
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

driver = GraphDatabase.driver("bolt://neo4j:7687", auth=("neo4j", "password"))

def fetch_graph_data(session):
    result = session.run("MATCH (n:Subscriber)-[r:CALL]->(m:Subscriber) RETURN n, r, m")
    nodes = []
    edges = []
    features = []
    node_map = {}
    idx = 0
    for record in result:
        n = record['n']
        m = record['m']
        r = record['r']
        if n['id'] not in node_map:
            node_map[n['id']] = idx
            nodes.append(n['id'])
            features.append(n['features'])
            idx += 1
        if m['id'] not in node_map:
            node_map[m['id']] = idx
            nodes.append(m['id'])
            features.append(m['features'])
            idx += 1
        edges.append([node_map[n['id']], node_map[m['id']]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

def reconstruction_loss(embeddings, edge_index):
    # Simple link prediction loss for unsupervised
    pos_loss = -torch.log(torch.sigmoid((embeddings[edge_index[0]] * embeddings[edge_index[1]]).sum(dim=1))).mean()
    neg_edge_index = negative_sampling(edge_index, embeddings.size(0))  # Need to import from torch_geometric.utils
    neg_loss = -torch.log(1 - torch.sigmoid((embeddings[neg_edge_index[0]] * embeddings[neg_edge_index[1]]).sum(dim=1))).mean()
    return pos_loss + neg_loss

with driver.session() as session:
    data = fetch_graph_data(session)

model = GraphSAGE(in_channels=4, hidden_channels=16, out_channels=8)
optimizer = Adam(model.parameters(), lr=0.01)

loader = NeighborLoader(data, num_neighbors=[10, 10], batch_size=128)

for epoch in range(100):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        embeddings = model(batch.x, batch.edge_index)
        loss = reconstruction_loss(embeddings, batch.edge_index)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    logger.info(f"Epoch {epoch}: Loss {total_loss / len(loader)}")

torch.save(model.state_dict(), 'models/graphsage.pt')
