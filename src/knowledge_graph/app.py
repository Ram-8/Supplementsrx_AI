import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from neo4j import GraphDatabase
from dotenv import load_dotenv

from fastapi.responses import JSONResponse

def run_safe(cypher: str, params: dict):
    try:
        with driver.session() as session:
            data = session.execute_read(run_query, cypher, params)
        return {"results": data}
    except Exception as e:
        # Return JSON instead of HTML 500 so clients don't crash on parsing
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "hint": "Check server logs for the Cypher line number.",
        })

# Load env
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7688")
NEO4J_USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "supplements_pass")

# Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# FastAPI app
app = FastAPI(title="Supplements KG Retrieval", version="1.0.0")

# CORS (dev-friendly; tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuery(BaseModel):
    intent: str  # SUPPLEMENTS_FOR_CONDITION | RISKS_OF_SUPPLEMENT | DOSAGE_FOR_SUPPLEMENT_CONDITION
    supplement: Optional[str] = None
    condition: Optional[str] = None
    drug: Optional[str] = None
    limit: int = 20

def run_query(tx, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = tx.run(cypher, **params)
    return [r.data() for r in result]

@app.get("/health")
def health():
    try:
        with driver.session() as s:
            s.run("RETURN 1").single()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/retrieve")
def retrieve(q: UserQuery):
    limit = max(1, min(q.limit, 50))

    if q.intent == "SUPPLEMENTS_FOR_CONDITION" and q.condition:
        cypher = """
        WITH toLower($condition) AS q
        CALL db.index.fulltext.queryNodes('nameSearch', $condition) YIELD node, score
        WITH node WHERE node:Condition AND toLower(node.name) CONTAINS q
        MATCH (s:Supplement)-[rel:MENTIONED_FOR|DOSAGE_GUIDELINE_FOR]->(node)
        OPTIONAL MATCH (s)-[i:INTERACTS_WITH]->(d:Drug)
        RETURN
          node.name AS condition,
          s.name     AS supplement,
          type(rel)  AS relationType,
          rel.dosage    AS dosage,
          rel.frequency AS frequency,
          rel.duration  AS duration,
          rel.form      AS form,
          rel.notes     AS notes,
          [x IN collect(DISTINCT {drug: d.name, severity: i.severity, description: i.description}) WHERE x.drug IS NOT NULL] AS interactions,
          s.source_url   AS source_url,
          s.last_updated AS last_updated
        ORDER BY s.name
        LIMIT $limit
        """
        params = {"condition": q.condition, "limit": limit}

    elif q.intent == "RISKS_OF_SUPPLEMENT" and q.supplement:
        cypher = """
        WITH toLower($supplement) AS q
        CALL db.index.fulltext.queryNodes('nameSearch', $supplement) YIELD node, score
        WITH node WHERE node:Supplement AND toLower(node.name) CONTAINS q
        OPTIONAL MATCH (node)-[i:INTERACTS_WITH]->(d:Drug)
        RETURN node.name AS supplement,
               [x IN collect({drug: d.name, severity: i.severity, description: i.description}) WHERE x.drug IS NOT NULL] AS interactions,
               node.source_url AS source_url,
               node.last_updated AS last_updated
        LIMIT 1
        """
        params = {"supplement": q.supplement}

    elif q.intent == "DOSAGE_FOR_SUPPLEMENT_CONDITION" and q.supplement and q.condition:
        cypher = """
        WITH toLower($supplement) AS sQ, toLower($condition) AS cQ
        MATCH (s:Supplement) WHERE toLower(s.name) CONTAINS sQ
        MATCH (c:Condition) WHERE toLower(c.name) CONTAINS cQ
        MATCH (s)-[r:DOSAGE_GUIDELINE_FOR]->(c)
        RETURN s.name AS supplement, c.name AS condition,
               r.dosage AS dosage, r.frequency AS frequency, r.duration AS duration, r.form AS form, r.notes AS notes,
               s.source_url AS source_url, s.last_updated AS last_updated
        ORDER BY s.name
        LIMIT $limit
        """
        params = {"supplement": q.supplement, "condition": q.condition, "limit": limit}

    else:
        return {"error": "Unsupported intent or missing fields. See docs for required fields per intent."}

    with driver.session() as session:
        data = session.execute_read(run_query, cypher, params)
    return {"results": data}

@app.post("/retrieve_v2")
def retrieve_v2(q: UserQuery):
    """
    Cleaner output:
    - Filter interactions by q.drug (e.g., "metformin"), if provided
    - Collect dosage guidelines into a list
    - Trim long descriptions
    - Sort interactions by severity (High > Moderate > Minor)
    """
    limit = max(1, min(q.limit, 50))

    if q.intent == "SUPPLEMENTS_FOR_CONDITION" and q.condition:
        cypher = """
        WITH toLower($condition) AS q, toLower(coalesce($drug, "")) AS drugQ
        CALL db.index.fulltext.queryNodes('nameSearch', $condition) YIELD node, score
        WITH node, drugQ
        WHERE node:Condition AND toLower(node.name) CONTAINS toLower($condition)

        // Base relation: mentioned or dosage-linked to the condition
        MATCH (s:Supplement)-[rel:MENTIONED_FOR|DOSAGE_GUIDELINE_FOR]->(node)

        // Collect any dosage guidelines for (s,node)
        OPTIONAL MATCH (s)-[drel:DOSAGE_GUIDELINE_FOR]->(node)
        WITH node, s, rel, collect(DISTINCT {
            dosage: drel.dosage, frequency: drel.frequency, duration: drel.duration,
            form: drel.form, notes: drel.notes
        }) AS dosage_list, drugQ

        // Optional interactions, filter by drug if provided
        OPTIONAL MATCH (s)-[i:INTERACTS_WITH]->(d:Drug)
        WITH node, s, rel, dosage_list, i, d, drugQ,
             CASE toUpper(i.severity)
                  WHEN 'HIGH' THEN 3
                  WHEN 'MODERATE' THEN 2
                  WHEN 'MINOR' THEN 1
                  ELSE 0
             END AS sev
        WHERE d IS NULL OR drugQ = "" OR toLower(d.name) CONTAINS drugQ

        WITH node, s, rel, dosage_list,
             [x IN collect(DISTINCT {
                 drug: d.name,
                 severity: i.severity,
                 description: substring(i.description, 0, 320),
                 severityScore: sev
             }) WHERE x.drug IS NOT NULL
             ] AS inter_raw

        WITH node, s, rel, dosage_list,
             [x IN inter_raw ORDER BY x.severityScore DESC][0..5] AS interactions

        RETURN
          node.name AS condition,
          s.name    AS supplement,
          type(rel) AS relationType,   // "MENTIONED_FOR" or "DOSAGE_GUIDELINE_FOR"
          dosage_list                  AS dosage_guidelines,
          interactions                 AS interactions,
          s.source_url                 AS source_url,
          s.last_updated               AS last_updated
        ORDER BY s.name
        LIMIT $limit
        """
        params = {"condition": q.condition, "drug": q.drug, "limit": limit}

    elif q.intent == "RISKS_OF_SUPPLEMENT" and q.supplement:
        cypher = """
        WITH toLower($supplement) AS q, toLower(coalesce($drug,"")) AS drugQ
        CALL db.index.fulltext.queryNodes('nameSearch', $supplement) YIELD node, score
        WITH node, drugQ WHERE node:Supplement AND toLower(node.name) CONTAINS q

        OPTIONAL MATCH (node)-[i:INTERACTS_WITH]->(d:Drug)
        WITH node, i, d, drugQ,
             CASE toUpper(i.severity)
                  WHEN 'HIGH' THEN 3
                  WHEN 'MODERATE' THEN 2
                  WHEN 'MINOR' THEN 1
                  ELSE 0
             END AS sev
        WHERE d IS NULL OR drugQ = "" OR toLower(d.name) CONTAINS drugQ

        WITH node,
             [x IN collect({
                 drug: d.name,
                 severity: i.severity,
                 description: substring(i.description, 0, 320),
                 severityScore: sev
             }) WHERE x.drug IS NOT NULL] AS inter_raw

        RETURN node.name AS supplement,
               [x IN inter_raw ORDER BY x.severityScore DESC][0..10] AS interactions,
               node.source_url AS source_url,
               node.last_updated AS last_updated
        LIMIT 1
        """
        params = {"supplement": q.supplement, "drug": q.drug}

    elif q.intent == "DOSAGE_FOR_SUPPLEMENT_CONDITION" and q.supplement and q.condition:
        cypher = """
        WITH toLower($supplement) AS sQ, toLower($condition) AS cQ
        MATCH (s:Supplement) WHERE toLower(s.name) CONTAINS sQ
        MATCH (c:Condition)  WHERE toLower(c.name) CONTAINS cQ
        MATCH (s)-[r:DOSAGE_GUIDELINE_FOR]->(c)
        RETURN s.name AS supplement, c.name AS condition,
               collect(DISTINCT {
                 dosage: r.dosage, frequency: r.frequency, duration: r.duration, form: r.form, notes: r.notes
               }) AS dosage_guidelines,
               s.source_url AS source_url, s.last_updated AS last_updated
        ORDER BY s.name
        LIMIT $limit
        """
        params = {"supplement": q.supplement, "condition": q.condition, "limit": limit}

    else:
        return {"error": "Unsupported intent or missing fields for /retrieve_v2."}

    with driver.session() as session:
        data = session.execute_read(run_query, cypher, params)
    return {"results": data}

@app.post("/retrieve_v3")
def retrieve_v3(q: UserQuery):
    """
    Safer retrieval (null-guards, no inline // comments in Cypher):
    - SUPPLEMENTS_FOR_CONDITION optionally filters interactions by q.drug
    - RISKS_OF_SUPPLEMENT optionally filters by q.drug
    - DOSAGE_FOR_SUPPLEMENT_CONDITION returns grouped dosage guidelines
    """
    limit = max(1, min(q.limit, 50))

    if q.intent == "SUPPLEMENTS_FOR_CONDITION" and q.condition:
        cypher = """
        WITH toLower($condition) AS condQ, toLower(coalesce($drug,'')) AS drugQ
        MATCH (c:Condition) WHERE toLower(c.name) CONTAINS condQ
        MATCH (s:Supplement)-[rel:MENTIONED_FOR|DOSAGE_GUIDELINE_FOR]->(c)
        OPTIONAL MATCH (s)-[dg:DOSAGE_GUIDELINE_FOR]->(c)
        WITH c, s, rel,
             collect(DISTINCT {
               dosage: dg.dosage, frequency: dg.frequency, duration: dg.duration,
               form: dg.form, notes: dg.notes
             }) AS dosage_list,
             drugQ
        OPTIONAL MATCH (s)-[i:INTERACTS_WITH]->(d:Drug)
        WITH c, s, rel, dosage_list, i, d, drugQ
        WHERE d IS NULL OR drugQ = '' OR toLower(d.name) CONTAINS drugQ
        WITH c, s, rel, dosage_list,
             [x IN collect(DISTINCT {
               drug: coalesce(d.name,''),
               severity: coalesce(i.severity,''),
               description: substring(coalesce(i.description,''), 0, 320),
               severityScore: CASE toUpper(coalesce(i.severity,'')) WHEN 'HIGH' THEN 3 WHEN 'MODERATE' THEN 2 WHEN 'MINOR' THEN 1 ELSE 0 END
             }) WHERE x.drug <> '' ] AS interactions
        RETURN
          c.name AS condition,
          s.name AS supplement,
          type(rel) AS relationType,
          dosage_list AS dosage_guidelines,
          interactions AS interactions,
          s.source_url AS source_url,
          s.last_updated AS last_updated
        ORDER BY s.name
        LIMIT $limit
        """
        params = {"condition": q.condition, "drug": q.drug, "limit": limit}

    elif q.intent == "RISKS_OF_SUPPLEMENT" and q.supplement:
        cypher = """
        WITH toLower($supplement) AS suppQ, toLower(coalesce($drug,'')) AS drugQ
        MATCH (s:Supplement) WHERE toLower(s.name) CONTAINS suppQ
        OPTIONAL MATCH (s)-[i:INTERACTS_WITH]->(d:Drug)
        WITH s, i, d, drugQ
        WHERE d IS NULL OR drugQ = '' OR toLower(d.name) CONTAINS drugQ
        WITH s,
             [x IN collect(DISTINCT {
               drug: coalesce(d.name,''),
               severity: coalesce(i.severity,''),
               description: substring(coalesce(i.description,''), 0, 320),
               severityScore: CASE toUpper(coalesce(i.severity,'')) WHEN 'HIGH' THEN 3 WHEN 'MODERATE' THEN 2 WHEN 'MINOR' THEN 1 ELSE 0 END
             }) WHERE x.drug <> '' ] AS inter_raw
        RETURN s.name AS supplement,
               [x IN inter_raw ORDER BY x.severityScore DESC][0..10] AS interactions,
               s.source_url AS source_url,
               s.last_updated AS last_updated
        LIMIT 1
        """
        params = {"supplement": q.supplement, "drug": q.drug}

    elif q.intent == "DOSAGE_FOR_SUPPLEMENT_CONDITION" and q.supplement and q.condition:
        cypher = """
        WITH toLower($supplement) AS sQ, toLower($condition) AS cQ
        MATCH (s:Supplement) WHERE toLower(s.name) CONTAINS sQ
        MATCH (c:Condition)  WHERE toLower(c.name) CONTAINS cQ
        MATCH (s)-[r:DOSAGE_GUIDELINE_FOR]->(c)
        RETURN s.name AS supplement, c.name AS condition,
               collect(DISTINCT {
                 dosage: r.dosage, frequency: r.frequency, duration: r.duration, form: r.form, notes: r.notes
               }) AS dosage_guidelines,
               s.source_url AS source_url, s.last_updated AS last_updated
        ORDER BY s.name
        LIMIT $limit
        """
        params = {"supplement": q.supplement, "condition": q.condition, "limit": limit}

    else:
        return {"error": "Unsupported intent or missing fields for /retrieve_v3."}

    with driver.session() as session:
        data = session.execute_read(run_query, cypher, params)
    return {"results": data}

DRUG_SYNONYM_MAP = {
    "metformin": ["metformin", "antidiabetes", "antidiabetic", "biguanide"],
    "glipizide": ["glipizide", "antidiabetes", "antidiabetic", "sulfonylurea"],
    "insulin": ["insulin", "antidiabetes", "antidiabetic"],
    "warfarin": ["warfarin", "anticoagulant", "antiplatelet"],
    "clopidogrel": ["clopidogrel", "antiplatelet", "anticoagulant"],
    "levothyroxine": ["levothyroxine", "thyroid hormone", "thyroid"],
    "doxorubicin": ["doxorubicin", "antitumor antibiotics", "anthracycline"],
    "cyclophosphamide": ["cyclophosphamide", "alkylating agents", "alkylating"],
}

def keywords_for_drug(s: str | None) -> list[str]:
    if not s:
        return []
    s = s.lower().strip()
    base = [s]
    extra = DRUG_SYNONYM_MAP.get(s, [])
    # de-dupe, keep order
    seen, out = set(), []
    for k in base + extra:
        if k not in seen and k:
            seen.add(k); out.append(k)
    return out


@app.post("/retrieve_v4")
def retrieve_v4(q: UserQuery):
    """
    Safer & smarter retrieval:
    - Filters interactions by drug *or* drug class via keyword matching (e.g., 'metformin' → 'antidiabetes')
    - Sorts interactions by severity using UNWIND (no ORDER BY inside list comprehension)
    - Returns structured dosage; if none, includes supplement-level dosing_text fallback
    """
    limit = max(1, min(q.limit, 50))
    drug_keywords = [k.lower() for k in keywords_for_drug(q.drug)]

    if q.intent == "SUPPLEMENTS_FOR_CONDITION" and q.condition:
        cypher = """
        WITH toLower($condition) AS condQ, $drugKeywords AS kws
        MATCH (c:Condition) WHERE toLower(c.name) CONTAINS condQ
        MATCH (s:Supplement)-[rel:MENTIONED_FOR|DOSAGE_GUIDELINE_FOR]->(c)
        OPTIONAL MATCH (s)-[dg:DOSAGE_GUIDELINE_FOR]->(c)
        WITH c, s, rel,
             collect(DISTINCT {
               dosage: dg.dosage, frequency: dg.frequency, duration: dg.duration,
               form: dg.form, notes: dg.notes
             }) AS dosage_list, kws

        OPTIONAL MATCH (s)-[i:INTERACTS_WITH]->(d:Drug)
        WITH c, s, rel, dosage_list, i, d, kws
        WHERE d IS NULL OR size(kws)=0 OR ANY(kw IN kws WHERE toLower(d.name) CONTAINS kw)

        WITH c, s, rel, dosage_list,
             collect(DISTINCT {
               drug: coalesce(d.name,''),
               severity: coalesce(i.severity,''),
               description: substring(coalesce(i.description,''), 0, 320),
               severityScore: CASE toUpper(coalesce(i.severity,'')) WHEN 'HIGH' THEN 3 WHEN 'MODERATE' THEN 2 WHEN 'MINOR' THEN 1 ELSE 0 END
             }) AS inter_raw

        UNWIND inter_raw AS x
        WITH c, s, rel, dosage_list, x
        ORDER BY x.severityScore DESC
        WITH c, s, rel, dosage_list, collect(x)[0..5] AS interactions

        RETURN
          c.name AS condition,
          s.name AS supplement,
          type(rel) AS relationType,
          dosage_list AS dosage_guidelines,
          interactions AS interactions,
          s.source_url AS source_url,
          s.last_updated AS last_updated
        ORDER BY s.name
        LIMIT $limit
        """
        params = {"condition": q.condition, "drugKeywords": drug_keywords, "limit": limit}

    elif q.intent == "RISKS_OF_SUPPLEMENT" and q.supplement:
        cypher = """
        WITH toLower($supplement) AS suppQ, $drugKeywords AS kws
        MATCH (s:Supplement) WHERE toLower(s.name) CONTAINS suppQ
        OPTIONAL MATCH (s)-[i:INTERACTS_WITH]->(d:Drug)
        WITH s, i, d, kws
        WHERE d IS NULL OR size(kws)=0 OR ANY(kw IN kws WHERE toLower(d.name) CONTAINS kw)

        WITH s,
             collect(DISTINCT {
               drug: coalesce(d.name,''),
               severity: coalesce(i.severity,''),
               description: substring(coalesce(i.description,''), 0, 320),
               severityScore: CASE toUpper(coalesce(i.severity,'')) WHEN 'HIGH' THEN 3 WHEN 'MODERATE' THEN 2 WHEN 'MINOR' THEN 1 ELSE 0 END
             }) AS inter_raw

        UNWIND inter_raw AS x
        WITH s, x
        ORDER BY x.severityScore DESC
        WITH s, collect(x)[0..10] AS interactions

        RETURN s.name AS supplement,
               interactions,
               s.source_url AS source_url,
               s.last_updated AS last_updated
        LIMIT 1
        """
        params = {"supplement": q.supplement, "drugKeywords": drug_keywords}

    elif q.intent == "DOSAGE_FOR_SUPPLEMENT_CONDITION" and q.supplement and q.condition:
        cypher = """
        WITH toLower($supplement) AS sQ, toLower($condition) AS cQ
        MATCH (s:Supplement) WHERE toLower(s.name) CONTAINS sQ
        MATCH (c:Condition)  WHERE toLower(c.name) CONTAINS cQ
        OPTIONAL MATCH (s)-[r:DOSAGE_GUIDELINE_FOR]->(c)
        WITH s, c, collect(DISTINCT {
             dosage: r.dosage, frequency: r.frequency, duration: r.duration, form: r.form, notes: r.notes
        }) AS dg

        WITH s, c, [x IN dg WHERE coalesce(x.dosage,'') <> '' OR coalesce(x.frequency,'') <> '' OR
                                coalesce(x.duration,'') <> '' OR coalesce(x.form,'') <> '' OR
                                coalesce(x.notes,'') <> ''] AS clean_dg

        RETURN s.name AS supplement, c.name AS condition,
               CASE WHEN size(clean_dg) > 0
                    THEN clean_dg
                    ELSE [ { notes: coalesce(s.dosing_text, 'No structured dosage available in KG') } ]
               END AS dosage_guidelines,
               s.source_url AS source_url, s.last_updated AS last_updated
        LIMIT $limit
        """
        params = {"supplement": q.supplement, "condition": q.condition, "limit": limit}

    else:
        return {"error": "Unsupported intent or missing fields for /retrieve_v4."}

    with driver.session() as session:
        data = session.execute_read(run_query, cypher, params)
    return {"results": data}
@app.post("/retrieve_v5")
def retrieve_v5(q: UserQuery):
    limit = max(1, min(q.limit, 50))
    drug_keywords = [k.lower() for k in keywords_for_drug(q.drug)]

    if q.intent == "SUPPLEMENTS_FOR_CONDITION" and q.condition:
        cypher = """
        WITH $condition AS term, $drugKeywords AS kws, $limit AS lim
        CALL db.index.fulltext.queryNodes('nameSearch', term) YIELD node, score
        WITH node, kws, lim
        WHERE node:Condition
        MATCH (s:Supplement)-[rel:MENTIONED_FOR|DOSAGE_GUIDELINE_FOR]->(node)
        OPTIONAL MATCH (s)-[dg:DOSAGE_GUIDELINE_FOR]->(node)
        WITH node, s, rel, kws, lim,
             [x IN collect(DISTINCT {
                 dosage: dg.dosage, frequency: dg.frequency, duration: dg.duration, form: dg.form, notes: dg.notes
             }) WHERE coalesce(x.dosage,'') <> '' OR coalesce(x.frequency,'') <> '' OR coalesce(x.duration,'') <> '' OR coalesce(x.form,'') <> '' OR coalesce(x.notes,'') <> ''
             ] AS dosage_list

        OPTIONAL MATCH (s)-[i:INTERACTS_WITH]->(d:Drug)
        WITH node, s, rel, dosage_list, i, d, kws, lim
        WHERE d IS NULL OR size(kws)=0 OR ANY(kw IN kws WHERE toLower(d.name) CONTAINS kw)

        WITH node, s, rel, dosage_list,
             collect(DISTINCT {
               drug: coalesce(d.name,''),
               severity: coalesce(i.severity,''),
               description: substring(coalesce(i.description,''), 0, 320),
               severityScore: CASE toUpper(coalesce(i.severity,'')) WHEN 'HIGH' THEN 3 WHEN 'MODERATE' THEN 2 WHEN 'MINOR' THEN 1 ELSE 0 END
             }) AS inter_raw, lim

        UNWIND inter_raw AS x
        WITH node, s, rel, dosage_list, x, lim
        ORDER BY x.severityScore DESC
        WITH node, s, rel, dosage_list, collect(x)[0..5] AS interactions, lim

        RETURN
          node.name AS condition,
          s.name    AS supplement,
          type(rel) AS relationType,
          CASE WHEN size(dosage_list) > 0 THEN dosage_list ELSE
               [ { notes: coalesce(s.dosing_text, 'No structured dosage available in KG') } ]
          END AS dosage_guidelines,
          interactions AS interactions,
          s.source_url AS source_url,
          s.last_updated AS last_updated
        ORDER BY s.name
        LIMIT $limit
        """
        params = {"condition": q.condition, "drugKeywords": drug_keywords, "limit": limit}

    elif q.intent == "RISKS_OF_SUPPLEMENT" and q.supplement:
        cypher = """
        WITH $supplement AS term, $drugKeywords AS kws
        CALL db.index.fulltext.queryNodes('nameSearch', term) YIELD node, score
        WITH node, kws WHERE node:Supplement
        OPTIONAL MATCH (node)-[i:INTERACTS_WITH]->(d:Drug)
        WHERE d IS NULL OR size(kws)=0 OR ANY(kw IN kws WHERE toLower(d.name) CONTAINS kw)

        WITH node,
             collect(DISTINCT {
               drug: coalesce(d.name,''),
               severity: coalesce(i.severity,''),
               description: substring(coalesce(i.description,''), 0, 320),
               severityScore: CASE toUpper(coalesce(i.severity,'')) WHEN 'HIGH' THEN 3 WHEN 'MODERATE' THEN 2 WHEN 'MINOR' THEN 1 ELSE 0 END
             }) AS inter_raw

        UNWIND inter_raw AS x
        WITH node, x
        ORDER BY x.severityScore DESC
        RETURN node.name AS supplement,
               collect(x)[0..10] AS interactions,
               node.source_url AS source_url,
               node.last_updated AS last_updated
        LIMIT 1
        """
        params = {"supplement": q.supplement, "drugKeywords": drug_keywords}

    elif q.intent == "DOSAGE_FOR_SUPPLEMENT_CONDITION" and q.supplement and q.condition:
        cypher = """
        WITH $supplement AS sTerm, $condition AS cTerm, $limit AS lim
        CALL db.index.fulltext.queryNodes('nameSearch', sTerm) YIELD node AS s, score
        WITH s, cTerm, lim WHERE s:Supplement
        CALL db.index.fulltext.queryNodes('nameSearch', cTerm) YIELD node AS c, score
        WITH s, c, lim WHERE c:Condition
        OPTIONAL MATCH (s)-[r:DOSAGE_GUIDELINE_FOR]->(c)
        WITH s, c, lim,
             [x IN collect(DISTINCT {
                 dosage: r.dosage, frequency: r.frequency, duration: r.duration, form: r.form, notes: r.notes
             }) WHERE coalesce(x.dosage,'') <> '' OR coalesce(x.frequency,'') <> '' OR coalesce(x.duration,'') <> '' OR coalesce(x.form,'') <> '' OR coalesce(x.notes,'') <> ''
             ] AS clean_dg

        RETURN s.name AS supplement, c.name AS condition,
               CASE WHEN size(clean_dg) > 0
                    THEN clean_dg
                    ELSE [ { notes: coalesce(s.dosing_text, 'No structured dosage available in KG') } ]
               END AS dosage_guidelines,
               s.source_url AS source_url, s.last_updated AS last_updated
        LIMIT $limit
        """
        params = {"supplement": q.supplement, "condition": q.condition, "limit": limit}

    else:
        return {"error": "Unsupported intent or missing fields for /retrieve_v5."}

    with driver.session() as session:
        data = session.execute_read(run_query, cypher, params)
    return {"results": data}
@app.post("/retrieve_v6")
def retrieve_v6(q: UserQuery):
    """
    Unified, robust retrieval:
    - Full-text name matching across Supplement/Condition
    - Drug filter accepts brand/generic and matches classes via keywords_for_drug()
    - Interactions sorted by severity in a subquery (safe even when empty)
    - Dosage guidelines with fallback to s.dosing_text
    """
    lim = max(1, min(q.limit, 50))
    kws = [k.lower() for k in keywords_for_drug(q.drug)]

    if q.intent == "SUPPLEMENTS_FOR_CONDITION" and q.condition:
        cypher = """
        WITH $condition AS term, $drugKeywords AS kws, $limit AS lim
        CALL db.index.fulltext.queryNodes('nameSearch', term) YIELD node, score
        WITH node, kws, lim
        WHERE node:Condition

        MATCH (s:Supplement)-[rel:MENTIONED_FOR|DOSAGE_GUIDELINE_FOR]->(node)

        OPTIONAL MATCH (s)-[dg:DOSAGE_GUIDELINE_FOR]->(node)
        WITH node, s, rel, kws, lim,
             [x IN collect(DISTINCT {
                 dosage: dg.dosage, frequency: dg.frequency, duration: dg.duration, form: dg.form, notes: dg.notes
             })
              WHERE ANY(v IN [x.dosage,x.frequency,x.duration,x.form,x.notes]
                        WHERE v IS NOT NULL AND v <> '')
             ] AS dosage_list

        OPTIONAL MATCH (s)-[i:INTERACTS_WITH]->(d:Drug)
        WITH node, s, rel, dosage_list, i, d, kws, lim
        WHERE d IS NULL OR size(kws)=0 OR ANY(kw IN kws WHERE toLower(d.name) CONTAINS kw)

        WITH node, s, rel, dosage_list, lim,
             collect(DISTINCT {
               drug: coalesce(d.name,''),
               severity: coalesce(i.severity,''),
               description: substring(coalesce(i.description,''), 0, 320),
               severityScore: CASE toUpper(coalesce(i.severity,'')) WHEN 'HIGH' THEN 3 WHEN 'MODERATE' THEN 2 WHEN 'MINOR' THEN 1 ELSE 0 END
             }) AS inter_raw

        CALL {
          WITH inter_raw
          UNWIND inter_raw AS x
          WITH x ORDER BY x.severityScore DESC
          RETURN collect(x)[0..5] AS interactions
        }

        RETURN
          node.name AS condition,
          s.name    AS supplement,
          type(rel) AS relationType,
          CASE WHEN size(dosage_list) > 0
               THEN dosage_list
               ELSE [ { notes: coalesce(s.dosing_text, 'No structured dosage available in KG') } ]
          END       AS dosage_guidelines,
          interactions            AS interactions,
          s.source_url            AS source_url,
          s.last_updated          AS last_updated
        ORDER BY s.name
        LIMIT $limit
        """
        params = {"condition": q.condition, "drugKeywords": kws, "limit": lim}
        return run_safe(cypher, params)

    elif q.intent == "RISKS_OF_SUPPLEMENT" and q.supplement:
        cypher = """
        WITH $supplement AS term, $drugKeywords AS kws
        CALL db.index.fulltext.queryNodes('nameSearch', term) YIELD node, score
        WITH node, kws WHERE node:Supplement
        OPTIONAL MATCH (node)-[i:INTERACTS_WITH]->(d:Drug)
        WHERE d IS NULL OR size(kws)=0 OR ANY(kw IN kws WHERE toLower(d.name) CONTAINS kw)

        WITH node,
             collect(DISTINCT {
               drug: coalesce(d.name,''),
               severity: coalesce(i.severity,''),
               description: substring(coalesce(i.description,''), 0, 320),
               severityScore: CASE toUpper(coalesce(i.severity,'')) WHEN 'HIGH' THEN 3 WHEN 'MODERATE' THEN 2 WHEN 'MINOR' THEN 1 ELSE 0 END
             }) AS inter_raw

        CALL {
          WITH inter_raw
          UNWIND inter_raw AS x
          WITH x ORDER BY x.severityScore DESC
          RETURN collect(x)[0..10] AS interactions
        }

        RETURN node.name AS supplement,
               interactions,
               node.source_url AS source_url,
               node.last_updated AS last_updated
        LIMIT 1
        """
        params = {"supplement": q.supplement, "drugKeywords": kws}
        return run_safe(cypher, params)

    elif q.intent == "DOSAGE_FOR_SUPPLEMENT_CONDITION" and q.supplement and q.condition:
        cypher = """
        WITH $supplement AS sTerm, $condition AS cTerm, $limit AS lim
        CALL db.index.fulltext.queryNodes('nameSearch', sTerm) YIELD node AS s, score
        WITH s, cTerm, lim WHERE s:Supplement
        CALL db.index.fulltext.queryNodes('nameSearch', cTerm) YIELD node AS c, score
        WITH s, c, lim WHERE c:Condition

        OPTIONAL MATCH (s)-[r:DOSAGE_GUIDELINE_FOR]->(c)
        WITH s, c, lim,
             [x IN collect(DISTINCT {
                 dosage: r.dosage, frequency: r.frequency, duration: r.duration, form: r.form, notes: r.notes
             })
              WHERE ANY(v IN [x.dosage,x.frequency,x.duration,x.form,x.notes]
                        WHERE v IS NOT NULL AND v <> '')
             ] AS clean_dg

        RETURN s.name AS supplement, c.name AS condition,
               CASE WHEN size(clean_dg) > 0
                    THEN clean_dg
                    ELSE [ { notes: coalesce(s.dosing_text, 'No structured dosage available in KG') } ]
               END AS dosage_guidelines,
               s.source_url AS source_url, s.last_updated AS last_updated
        LIMIT $limit
        """
        params = {"supplement": q.supplement, "condition": q.condition, "limit": lim}
        return run_safe(cypher, params)

    else:
        return {"error": "Unsupported intent or missing fields for /retrieve_v6."}
