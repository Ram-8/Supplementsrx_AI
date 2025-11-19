import os, json, glob
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
URI = os.getenv("NEO4J_URI", "bolt://localhost:7688")
USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "supplements_pass")

# Default to processed data directory if DATA_DIR not set
if os.getenv("DATA_DIR"):
    DATA_DIR = os.getenv("DATA_DIR")
else:
    # Default to data/processed relative to project root
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = str(PROJECT_ROOT / "data" / "processed")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def norm(x): return x.strip() if isinstance(x, str) else x

def load_one(tx, data):
    name = norm(data.get("supplement_name"))
    if not name: return

    # Supplement node (with provenance)
    tx.run("""
        MERGE (s:Supplement {name: $name})
        SET s.scientific_name      = $scientific_name,
            s.overview_text        = $overview_text,
            s.effectiveness_text   = $effectiveness_text,
            s.safety_text          = $safety_text,
            s.dosing_text          = $dosing_text,
            s.interactions_text    = $interactions_text,
            s.mechanism_text       = $mechanism_text,
            s.general_safety       = $general_safety,
            s.pregnancy_safety     = $pregnancy_safety,
            s.breastfeeding_safety = $breastfeeding_safety,
            s.children_safety      = $children_safety,
            s.source_url           = $source_url,
            s.data_version         = $data_version,
            s.last_updated         = $last_updated
    """,
    name=name,
    scientific_name=norm(data.get("scientific_name")),
    overview_text=norm(data.get("overview_text")),
    effectiveness_text=norm(data.get("effectiveness_text")),
    safety_text=norm(data.get("safety_text")),
    dosing_text=norm(data.get("dosing_text")),
    interactions_text=norm(data.get("interactions_text")),
    mechanism_text=norm(data.get("mechanism_text")),
    general_safety=norm(data.get("safety_ratings",{}).get("general_safety")),
    pregnancy_safety=norm(data.get("safety_ratings",{}).get("pregnancy_safety")),
    breastfeeding_safety=norm(data.get("safety_ratings",{}).get("breastfeeding_safety")),
    children_safety=norm(data.get("safety_ratings",{}).get("children_safety")),
    source_url=norm(data.get("metadata",{}).get("source_url")),
    data_version=norm(data.get("metadata",{}).get("data_version")),
    last_updated=norm(data.get("metadata",{}).get("last_updated"))
    )

    # Conditions
    for cond in (data.get("conditions") or []):
        cond = norm(cond)
        if not cond: continue
        tx.run("""
            MATCH (s:Supplement {name: $supp})
            MERGE (c:Condition {name: $cond})
            MERGE (s)-[:MENTIONED_FOR]->(c)
        """, supp=name, cond=cond)

    # Drug Interactions
    for di in (data.get("drug_interactions") or []):
        drug_name = norm(di.get("drug_name"))
        if not drug_name: continue
        tx.run("""
            MATCH (s:Supplement {name: $supp})
            MERGE (d:Drug {name: $drug})
              ON CREATE SET d.drug_class = $drug_class
            MERGE (s)-[r:INTERACTS_WITH]->(d)
            SET r.severity        = $severity,
                r.interaction_type= $interaction_type,
                r.description     = $description
        """,
        supp=name,
        drug=drug_name,
        drug_class=norm(di.get("drug_class")),
        severity=norm(di.get("severity")),
        interaction_type=norm(di.get("interaction_type")),
        description=norm(di.get("description"))
        )

    # Dosage Guidelines (supplement ↔ condition)
    for dg in (data.get("dosage_guidelines") or []):
        cond = norm(dg.get("condition"))
        if not cond: continue
        tx.run("""
            MATCH (s:Supplement {name: $supp})
            MERGE (c:Condition {name: $cond})
            MERGE (s)-[r:DOSAGE_GUIDELINE_FOR]->(c)
            SET r.dosage    = $dosage,
                r.frequency = $frequency,
                r.duration  = $duration,
                r.form      = $form,
                r.notes     = $notes
        """,
        supp=name,
        cond=cond,
        dosage=norm(dg.get("dosage")),
        frequency=norm(dg.get("frequency")),
        duration=norm(dg.get("duration")),
        form=norm(dg.get("form")),
        notes=norm(dg.get("notes"))
        )

def main():
    if not DATA_DIR or not os.path.isdir(DATA_DIR):
        raise SystemExit(f"DATA_DIR not found: {DATA_DIR}")
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    print(f"Found {len(paths)} JSON files in {DATA_DIR}")
    with driver.session() as session:
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                session.execute_write(load_one, data)
                print(f"Loaded: {os.path.basename(p)}")
            except Exception as e:
                print(f"ERROR {p}: {e}")

if __name__ == "__main__":
    main()
    driver.close()