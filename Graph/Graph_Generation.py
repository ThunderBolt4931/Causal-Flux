import json
import networkx as nx
from neo4j import GraphDatabase
from intent_identifier import classify_intents
from dotenv import load_dotenv
import numpy as np
import os
import time
# from intents_plots import plot_driver_counts
# from number_intents_plots import plot_driver_occurrences
from Hierarcical_Retriver import HierarchicalRetriever

load_dotenv()

NEO4J_URI = "Your NEO4J_URI"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Your NEO4J_PASSWORD"

DATA_FILE = "Corpus\corpus.json"
CLUSTERINGS_FILE = "Clusters\clustered_transcripts.json"


retriver = HierarchicalRetriever(CLUSTERINGS_FILE)

with open(CLUSTERINGS_FILE, "r", encoding="utf-8") as f:
    CLUSTERINGS = json.load(f)

L1_CLUSTERS = {}
L2_CLUSTERS = []

DOC_TO_CLUSTERS: dict[str, dict[str, dict[str, set[str]]]] = {}

for c in CLUSTERINGS:
    ctype = c.get("type")
    if ctype == "L1Cluster":
        L1_CLUSTERS[c["id"]] = c

    elif ctype == "L2Cluster":
        L2_CLUSTERS.append(c)
        field = c["field"]          
        parent_id = c["parent_id"]  
        for tid in c.get("member_ids", []):  
            field_map = DOC_TO_CLUSTERS.setdefault(tid, {}).setdefault(
                field, {"L1": set(), "L2": set()}
            )
            field_map["L2"].add(c["id"])
            field_map["L1"].add(parent_id)


with open(DATA_FILE, "r", encoding="utf-8") as f:
    FULL_DATA = json.load(f)

ID_TO_TRANSCRIPT = {item["transcript_id"]: item for item in FULL_DATA}

DOMAIN_INTENTS = {
    "flight_domain", "insurance_domain", "hotel_domain",
    "retail_domain", "banking_domain", "telecom_domain"
}


class Neo4jGraphRAG:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def load_graph(self):
        print("Starting graph load into Neo4j...")
        start = time.time()
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Transcript) REQUIRE t.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:InteractionDriver) REQUIRE i.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:CallReason) REQUIRE r.name IS UNIQUE")

            batch_size = 500
            total = len(FULL_DATA)
            for i in range(0, total, batch_size):
                batch = FULL_DATA[i:i + batch_size]
                records = []
                for item in batch:
                    tid = item["transcript_id"]
                    domain = item.get("domain", "Unknown")
                    reason = item.get("intent", "Unknown")
                    
                    drivers_data = item.get("metadata", {}).get("predefined_interaction_drivers", [])
                    drivers = [d.get("driver") for d in drivers_data if "driver" in d]
                    
                    records.append({"id": tid, "domain": domain, "reason": reason, "drivers": drivers})

                session.run("""
                    UNWIND $records AS rec
                    MERGE (t:Transcript {id: rec.id})
                    MERGE (d:Domain {name: rec.domain})
                    MERGE (r:CallReason {name: rec.reason})
                    MERGE (t)-[:IN_DOMAIN]->(d)
                    MERGE (t)-[:HAS_REASON]->(r)
                    WITH rec, t
                    UNWIND rec.drivers AS name
                    MERGE (i:InteractionDriver {name: name})
                    MERGE (t)-[:HAS_DRIVER]->(i)
                """, records=records)
                print(f"Loaded {min(i + batch_size, total)} / {total} transcripts...")

        print(f"Graph loaded successfully in {time.time() - start:.2f} seconds.")

    def close(self):
        self.driver.close()

    def my_print(self, rank, score, tid, domain, reason_text, matched_intents, preview, clusters=None):
        print(f"\nRANK {rank} | PPR: {score:.6f}")
        print(f"ID     : {tid}")
        print(f"Domain : {domain}")
        print(f"Reason_for_call: {reason_text}")
        print(f"Drivers: {', '.join(matched_intents[:5])}{'...' if len(matched_intents) > 5 else ''}")

        if clusters:
            print("L1 Clusters (data-driven):")
            for field in ["reason_for_call", "summary", "keywords", "outcome"]:
                names = clusters.get(field, [])
                if names:
                    print(f"  - {field}: {', '.join(names)}")

        print(f"Preview: {preview}\n")
        print("-" * 100)


    def query(self, query_text: str, top_k: int = 10) -> list[dict]:
        query_intents = list(set(classify_intents(query_text)))

        domain_intents = {d for d in query_intents if d in DOMAIN_INTENTS}
        content_drivers = [d for d in query_intents if d not in DOMAIN_INTENTS]
        
        normalized_drivers = content_drivers + [d.replace(" ", "_") for d in content_drivers]

        allowed_domains = {
            d.split("_")[0].capitalize()
            for d in domain_intents
        }

        print(f"\n{'='*100}")
        print(f"QUERY: {query_text}")
        print(f"IDENTIFIED DRIVERS: {query_intents}")
        print(f"DOMAIN MASKING: {'ON -> ' + ', '.join(sorted(allowed_domains)) if allowed_domains else 'OFF -> All domains'}")
        print(f"{'='*100}\n")

        if allowed_domains :
            possible_nodes = retriver.retrieve(query_text, top_k_l1 = 6, top_k_l2_per_l1= 8,)['documents']
        else:
            possible_nodes = retriver.retrieve(query_text, top_k_l1 = 5, top_k_l2_per_l1= 7,)['documents']

        print(len(possible_nodes))
        if possible_nodes:
            possible_nodes = list(possible_nodes)
            print(f"APPLYING possible_nodes filter: {len(possible_nodes)} ids provided by caller")
        else:
            possible_nodes = None


        with self.driver.session() as session:
            where_clauses = []
            params = {"qdrivers": normalized_drivers}

            # base clause: driver names
            where_clauses.append("i.name IN $qdrivers")

            # optional domain clause
            if allowed_domains:
                where_clauses.append("EXISTS { MATCH (t)-[:IN_DOMAIN]->(d:Domain) WHERE d.name IN $domains }")
                params["domains"] = list(allowed_domains)

            # optional node-id filter (apply before aggregation)
            if possible_nodes:
                where_clauses.append("t.id IN $possible_nodes")
                params["possible_nodes"] = list(possible_nodes)

            cypher = """
                MATCH (t:Transcript)-[:HAS_DRIVER]->(i:InteractionDriver)
            """

            if where_clauses:
                cypher += " WHERE " + " AND ".join(where_clauses)

            cypher += """
                WITH t, count(*) AS overlap
                RETURN t.id AS id, overlap
                ORDER BY overlap DESC
            """

            result = session.run(cypher, **params)
            candidates = [r for r in result]


            if not candidates:
                print("No matches found after masking.")
                return []

            candidate_ids = [r["id"] for r in candidates]
            max_overlap = candidates[0]["overlap"]
            seed_ids = [r["id"] for r in candidates if r["overlap"] == max_overlap]

            print(f"Unmasked transcripts : {len(candidate_ids)}")
            print(f"Seeds (max overlap {max_overlap}) : {len(seed_ids)}")

            rels = session.run("""
                MATCH (t:Transcript)-[r:HAS_DRIVER|IN_DOMAIN|HAS_REASON]->(x)
                WHERE t.id IN $ids
                WITH t.id AS tid,
                     'Transcript' AS source_type,
                     labels(x)[0] AS target_type,
                     COALESCE(x.name, x.id) AS target_name
                RETURN tid, source_type, target_type, target_name
            """, ids=candidate_ids)

            G = nx.Graph()
            for r in rels:
                G.add_edge(("Transcript", r["tid"]), (r["target_type"], r["target_name"]))

            print(f"Subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

            personalization = {n: 0.0 for n in G.nodes()}
            for sid in seed_ids:
                personalization[("Transcript", sid)] = 1.0 / len(seed_ids)

            ppr = nx.pagerank(G,
                              alpha=0.85,
                              personalization=personalization,
                              max_iter=100
                              )
            scores = {k[1]: v for k, v in ppr.items() if k[0] == "Transcript"}
            top_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            print(f"\nTOP {len(top_results)} RESULTS")
            print("=" * 100)

            results = []
            retrieved_count = len(candidate_ids)

            for rank, (tid, score) in enumerate(top_results, 1):
                item = ID_TO_TRANSCRIPT[tid]
                domain = item.get("domain", "Unknown")
                reason = item.get("intent", "Unknown")
                
                drivers_data = item.get("metadata", {}).get("predefined_interaction_drivers", [])
                matched_drivers = [d.get("driver") for d in drivers_data if "driver" in d]

                conversation = []
                for turn in item.get("turns", []):
                    for u in turn.get("conversation", []):
                        speaker = u.get("speaker", "Unknown")
                        text = u.get("utterance", "")
                        conversation.append(f"{speaker}: {text}")
                preview = " | ".join(conversation[:])

                self.my_print(rank, score, tid, domain, reason, matched_drivers, preview)

                results.append({
                    "rank": rank,
                    "transcript_id": tid,
                    "ppr_score": round(score, 6),
                    "domain": domain,
                    "call_intent": reason,
                    "drivers": matched_drivers,
                    "driver_overlap": next((c["overlap"] for c in candidates if c["id"] == tid), 0),
                    "preview": preview,
                })
            results.append({
                "retrieved_count": f"There are {retrieved_count} calls in the knowledge base related to this query."
            })
            # plot_driver_counts(DATA_FILE, possible_nodes, query_intents)
            # plot_driver_occurrences(DATA_FILE, possible_nodes, query_intents)
            return results


    def clear_database(self):
        """Completely reset the database — works on Neo4j 5.x+ (AuraDB)."""
        print("CLEARING ENTIRE NEO4J DATABASE...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("All nodes and relationships deleted.")

            result = session.run("SHOW CONSTRAINTS YIELD name")
            constraint_names = [record["name"] for record in result]

            for name in constraint_names:
                session.run(f"DROP CONSTRAINT {name}")
                print(f"Dropped constraint: {name}")

            result = session.run("SHOW INDEXES YIELD name")
            index_names = [record["name"] for record in result]
            for name in index_names:
                session.run(f"DROP INDEX {name}")
                print(f"Dropped index: {name}")

        print("Database fully reset and ready for fresh data!\n")


if __name__ == "__main__":
    rag = Neo4jGraphRAG(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # If you have not built the graph yet, run once:
    # rag.clear_database()
    # rag.load_graph()

    start_time = time.time()
    # query = "How did the customer's report of a network outage in Florida lead to it being identified as a weather-related active outage and then cause the customer to feel frustrated?"
    query = "in the situations where the customer is not happy with the services compared to other brands, how do agents calm customerss down in situations of perceived churn?"
    result = rag.query(query, top_k=10)

    # print(f"\nFinal Results JSON:\n{json.dumps(result, indent=2)}")h

    rag.close()
    end_time = time.time()
    print("Time taken:", round(end_time - start_time, 2))