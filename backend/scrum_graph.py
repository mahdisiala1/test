from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass
import math
import json
import ast
import re

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage

class ScrumState(TypedDict, total=False):
    cahier_de_charge: str
    team: Dict[str, Any]
    validation_attempts: int
    max_validation_attempts: int

    cleaned_spec: str
    spec_cleaning: Dict[str, Any]      # info about cleaning steps
    spec_validation: Dict[str, Any]    # ok/errors

    requirements: List[Dict[str, Any]]
    product_backlog: List[Dict[str, Any]]
    refined_backlog: List[Dict[str, Any]]
    estimated_backlog: List[Dict[str, Any]]
    dependencies: List[Dict[str, Any]]

    sprint_backlogs: List[Dict[str, Any]]
    assignments: List[Dict[str, Any]]

    validation: Dict[str, Any]


model = "llama-3.1-8b-instant"
llm = ChatGroq(
    model=model,
    temperature=0
)


def llm_json(prompt: str) -> Dict[str, Any]:
    response = llm.invoke([
        HumanMessage(content=prompt + "\n\nReturn ONLY valid JSON. No markdown.")
    ])

    text = response.content.strip()

    # Optional: handle ```json blocks
    if text.find("```json") != -1:
        start_index = text.find("```json")
        end_index = text.find("```", start_index + 7)
        text = text[start_index + 7:end_index]
        print("new text is:" + text)

    return json.loads(text)


def parse_llm_json_or_python_dict(text: str):
    text = text.strip()

    # 1) try JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) try python dict safely
    try:
        return ast.literal_eval(text)
    except Exception:
        pass

    # 3) try extracting JSON block if model wrapped it
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return ast.literal_eval(candidate)
            except Exception:
                pass

    return None


def clean_specifications_llm_node(state: ScrumState) -> ScrumState:
    raw_spec = (state.get("cahier_de_charge") or "").strip()

    prompt = f"""
You are a senior business analyst.

Task:
Rewrite the following specification into a clean, structured version.

Rules:
- Keep the meaning the same
- Remove duplication
- Fix unclear sentences
- Output MUST be plain text (not JSON)
- Use this structure:

TITLE:
SCOPE:
ACTORS:
FUNCTIONAL REQUIREMENTS:
NON-FUNCTIONAL REQUIREMENTS:
CONSTRAINTS:
OUT OF SCOPE:
OPEN QUESTIONS:

Specification:
{raw_spec}
""".strip()

    resp = llm.invoke([HumanMessage(content=prompt)])

    state["cleaned_spec"] = resp.content.strip()
    state["spec_cleaning"] = {
        "ok": True,
        "method": "llm_rewrite",
        "model": model
    }
    print("cleaned_spec is:" + state["cleaned_spec"])
    return state


def validate_specifications_llm_node(state: ScrumState) -> ScrumState:
    spec = (state.get("cleaned_spec") or "").strip()
    print("cleaned_spec is:" + str(state.get("cleaned_spec")))

    prompt = f"""
Tu es un auditeur de cahier des charges.

Objectif:
Vérifier la QUALITÉ du cahier des charges.

Règles STRICTES:
1) Tu n'as PAS le droit de dire qu'une section est absente si elle existe.
2) Pour chaque erreur, tu dois fournir "evidence" = une citation exacte du texte.
3) Si tu ne peux pas citer, tu DOIS supprimer cette erreur.
4) Le document est en français. Les sections suivantes comptent:
   - "Contexte" ou "Objectifs" => scope
   - "Fonctionnalités attendues" => requirements
   - "Contraintes techniques" + "KPI" => NFR
   - "Équipe-projet" => team

Retourne UNIQUEMENT du JSON valide.
Utilise uniquement des guillemets doubles " ".

Schema:
{{
  "ok": true/false,
  "errors": [
    {{
      "code": "SCOPE_TOO_VAGUE|REQUIREMENTS_TOO_HIGH_LEVEL|NFR_TOO_WEAK|TEAM_INCOMPLETE|AMBIGUOUS|INCONSISTENT|OTHER",
      "message": "string (non vide)",
      "severity": "low|medium|high",
      "hint": "string",
      "evidence": "quote"
    }}
  ]
}}

Si le cahier des charges est globalement bon:
ok=true et errors=[]

Specification:
{spec}
""".strip()

    resp = llm_json(prompt)
    state["spec_validation"] = resp
    return state


def route_after_spec_validation(state: ScrumState) -> str:
    return "valid" if state["spec_validation"].get("ok") else "invalid"


def extract_requirements_node(state: ScrumState) -> ScrumState:
    spec = state.get("cleaned_spec", state["cahier_de_charge"])

    prompt = f"""
Extract requirements from this cahier de charge.

Return JSON with:
requirements: [
  {{
    "id": "R1",
    "type": "functional|nfr",
    "text": "...",
    "priority": "must|should|could",
    "notes": "optional"
  }}
]

SPEC:
{spec}
"""
    out = llm_json(prompt)
    print("requirement out is:" + str(out))
    state["requirements"] = out
    return state


def generate_product_backlog_node(state: ScrumState) -> ScrumState:
    reqs = state["requirements"]

    prompt = f"""
Convert requirements into a Scrum Product Backlog.
please for all the field use " " as separator and don't use ' '

Return JSON with:
product_backlog: [
  {{
    "epic": "Epic name",
    "stories": [
      {{
        "id": "US1",
        "title": "...",
        "as_a": "...",
        "i_want": "...",
        "so_that": "...",
        "acceptance_criteria": ["..."],
        "required_skills": ["backend","frontend","devops","qa"]
      }}
    ]
  }}
]

REQUIREMENTS:
{reqs}
"""
    out = llm_json(prompt)

    state["product_backlog"] = out["product_backlog"]
    return state


def refine_backlog_node(state: ScrumState) -> ScrumState:
    pb = state["product_backlog"]

    prompt = f"""
Refine the backlog using INVEST:
- split stories that are too big
- remove duplicates
- add missing acceptance criteria
Return JSON with:
refined_backlog: [
  {{
    "id": "US1",
    "title": "...",
    "description": "...",
    "acceptance_criteria": ["..."],
    "required_skills": ["backend","frontend","devops","qa"]
  }}
]

IMPORTANT:
- Output ONLY JSON.
- No python code.
- No markdown.
- No explanations.
PRODUCT_BACKLOG:
{pb}
"""
    out = llm_json(prompt)
    print("refined_backlog out is:" + str(out))
    state["refined_backlog"] = out["refined_backlog"]
    return state


def estimate_backlog_node(state: ScrumState) -> ScrumState:
    refined = state["product_backlog"]
    team = state["team"]

    if state.get("validation", {}).get("ok", False) == False:
        validation_feedback = state.get("validation")
    else:
        validation_feedback = {"ok": True}
    print("validation_feedback is:" + str(validation_feedback))

    prompt = f"""
Estimate each story using Fibonacci story points: 1,2,3,5,8,13,21.

Return JSON with:
{{
  "estimated_backlog": [
    {{
      "id": "...",
      "title": "...",
      "points": 1|2|3|5|8|13|21,
      "risk": "low|medium|high",
      "complexity": "low|medium|high",
      "required_skills": [...]
    }}
  ]
}}

IMPORTANT:
- Output ONLY JSON.
- No python code.
- No markdown.
- No explanations.

validationFeedback:
{validation_feedback}

TEAM:
{team}

STORIES:
{refined}
"""

    out = llm_json(prompt)
    print("estimated_backlog out is:" + str(out))
    state["estimated_backlog"] = out["estimated_backlog"]
    return state


def map_dependencies_node(state: ScrumState) -> ScrumState:
    stories = state["estimated_backlog"]

    prompt = f"""
Detect dependencies between stories.

Return JSON with:
dependencies: [
  {{
    "from": "US1",
    "to": "US5",
    "type": "blocks"
  }}
]
STORIES:
{stories}
"""
    out = llm_json(prompt)
    print("dependencies out is:" + str(out))
    state["dependencies"] = out["dependencies"]
    return state


def sprint_planner_node(state: ScrumState) -> ScrumState:
    stories = state["estimated_backlog"]
    deps = state.get("dependencies", [])
    capacity = int(state["team"].get("sprint_capacity_points", 20))

    # Simple heuristic: order by dependencies first (very simplified)
    # (In production: topological sort)
    ordered = stories[:]  # assume already prioritized by LLM

    sprints = []
    current = {"sprint": 1, "items": [], "total_points": 0}

    for st in ordered:
        pts = int(st["points"])
        if current["total_points"] + pts > capacity and current["items"]:
            sprints.append(current)
            current = {"sprint": current["sprint"] + 1, "items": [], "total_points": 0}

        current["items"].append(st["id"])
        current["total_points"] += pts

    if current["items"]:
        sprints.append(current)

    state["sprint_backlogs"] = sprints
    return state


def contributor_assigner_node(state: ScrumState) -> ScrumState:
    team_members = state["team"]["members"]
    stories = {s["id"]: s for s in state["estimated_backlog"]}

    # Very simple skill matching:
    # assign 1 main person who matches most skills
    assignments = []

    for sprint in state["sprint_backlogs"]:
        for story_id in sprint["items"]:
            story = stories[story_id]
            req_skills = set(story.get("required_skills", []))

            best = None
            best_score = -1

            for m in team_members:
                skills = set(m.get("skills", []))
                score = len(req_skills.intersection(skills))
                if score > best_score:
                    best_score = score
                    best = m

            assignments.append({
                "story_id": story_id,
                "title": story.get("title", ""),
                "assigned_to": best["name"] if best else None,
                "reason": f"matched_skills={best_score}"
            })

    state["assignments"] = assignments
    return state


def validation_node(state: ScrumState) -> ScrumState:
    state["validation_attempts"] = int(state.get("validation_attempts", 0)) + 1
    capacity = int(state["team"].get("sprint_capacity_points", 20))
    sprints = state["sprint_backlogs"]
    stories = {s["id"]: s for s in state["estimated_backlog"]}

    issues = []

    for sp in sprints:
        total = sum(int(stories[sid]["points"]) for sid in sp["items"])
        if total > capacity:
            issues.append({
                "type": "over_capacity",
                "sprint": sp["sprint"],
                "total_points": total,
                "capacity": capacity
            })

    # Example: detect too big stories
    for s in state["estimated_backlog"]:
        if int(s["points"]) >= 13:
            issues.append({
                "type": "story_too_big",
                "story_id": s["id"],
                "points": s["points"]
            })

    state["validation"] = {
        "ok": len(issues) == 0,
        "issues": issues
    }
    return state


def route_after_validation(state: ScrumState) -> str:
    attempts = int(state.get("validation_attempts", 0))
    max_attempts = int(state.get("max_validation_attempts", 3))

    if attempts >= max_attempts and not state["validation"]["ok"]:
        state["validation"]["stopped_reason"] = "Max validation attempts reached"

        return "stop"
    if state["validation"]["ok"]:
        print("validation ok, we can continue")
        return "done"
    # if stories too big -> refine again
    for issue in state["validation"]["issues"]:
        if issue["type"] == "story_too_big":
            print("we will refine because of issue:" + str(issue))
            return "refine"
    # if over capacity -> plan again
    print("we will plan again because of over capacity")
    return "replan"


def build_scrum_graph():
    g = StateGraph(ScrumState)
    g.add_node("clean_spec", clean_specifications_llm_node)
    g.add_node("validate_spec", validate_specifications_llm_node)
    g.add_node("extract_requirements", extract_requirements_node)
    g.add_node("generate_product_backlog", generate_product_backlog_node)
    g.add_node("refine_backlog", refine_backlog_node)
    g.add_node("estimate_backlog", estimate_backlog_node)
    g.add_node("map_dependencies", map_dependencies_node)
    g.add_node("sprint_planner", sprint_planner_node)
    g.add_node("contributor_assigner", contributor_assigner_node)
    g.add_node("validation", validation_node)

    # Edges
    g.set_entry_point("clean_spec")
    g.add_edge("clean_spec", "validate_spec")

    g.add_conditional_edges(
        "validate_spec",
        route_after_spec_validation,
        {
            "valid": "extract_requirements",
            "invalid": END
        }
    )

    g.add_edge("extract_requirements", "generate_product_backlog")
    g.add_edge("generate_product_backlog", "estimate_backlog")

    g.add_edge("refine_backlog", "estimate_backlog")
    g.add_edge("estimate_backlog", "map_dependencies")
    g.add_edge("map_dependencies", "sprint_planner")
    g.add_edge("sprint_planner", "contributor_assigner")
    g.add_edge("contributor_assigner", "validation")

    # Conditional routing after validation
    g.add_conditional_edges(
        "validation",
        route_after_validation,
        {
            "done": END,
            "refine": "estimate_backlog",
            "replan": "sprint_planner",
            "stop": END
        }
    )

    return g.compile()


if __name__ == "__main__":
    graph = build_scrum_graph()

    cahier_de_charge = """
Cahier des Charges : équipe Joy
Projet SummerCamp 2025
1. Contexte du projet
- Nom du projet :
PorterVision – Plateforme d’analyse stratégique automatisée basée sur les 5 forces de
Porter.
- Client :
● Direction de la stratégie
● Direction commerciale
● Cellules d’innovation ou transformation digitale
● Bureau d'Études
- Département concerné :
● Direction de la stratégie
● Département marketing et commercial
- Objectifs stratégiques :
● Digitaliser et centraliser les analyses de positionnement stratégique.
● Accélérer la prise de décision commerciale en fournissant des analyses automatisées
et visuelles.
● Intégrer l’IA dans les processus d’amélioration continue et de veille concurrentielle.
● Favoriser une culture data-driven dans les décisions stratégiques.
2. Problématique à résoudre
L’analyse des 5 forces de Porter est un outil stratégique clé pour évaluer la position
concurrentielle d’un projet ou d’une unité business. Cependant, ce processus est aujourd’hui
● manuel et chronophage, nécessitant un travail long d’analyse documentaire,
● non standardisé, avec des résultats variables selon les analystes,
● difficile à exploiter à grande échelle pour plusieurs projets en parallèle.
L’IA peut répondre à cette problématique en automatisant l’extraction, la classification et
l’analyse des informations stratégiques, afin d’apporter une vision rapide, cohérente et
exploitable de l’environnement concurrentiel.
3. Objectifs du projet
● Automatiser l’analyse stratégique des projets à partir de documents textuels grâce
à des techniques d’IA (NLP, classification).
● Prédire l’intensité de chaque force de Porter (forte, modérée, faible) selon les
données extraites.
● Classifier les informations collectées en fonction des 5 forces de Porter
(concurrence, nouveaux entrants, substitution, clients, fournisseurs).
● Générer une cartographie interactive des 5 forces pour chaque projet ou unité
analysée.
● Recommander des actions stratégiques basées sur l’analyse des forces :
différenciation, innovation, positionnement
4. Données disponibles
1. Data Sources:
● Project PDF Files : Descriptions of strategic projects projects(briefs, business plans,
PowerPoint presentations) provided by users. Access: Manual upload by users via
● a secure interface.
● Intelligent Web Scraping: Data extracted automatically from company websites,
LinkedIn, and other platforms. Access: Automated via AI agents with customizable filters.
● Internal Databases (optional): Integration of CRM, ERP, Trello, and internal documents
(Google Drive, SharePoint). Access: Integration via API with user-configurable
settings.
● External Market APIs :Market data (Statista, GDELT), industry trends, economic data
(Google Trends, CB Insights). Access: API key configuration by the administrator.
● Structured User Form Data entered via an intelligent assistant (guided form or interactive
chatbot). Access: Intuitive front-end interface with real-time validation.
DATASET SOURCES (For Agents)
● Crunchbase Startup and competitor information.
● CB Insights Market insights and company analytics.
● World Bank Open Data Economic and development data.
● Google Trends API Search trend data for market analysis.
● Common pretraining context Large-scale web data for AI model training.
● Gartner Reports Industry research and strategic insights.
● Twitter/X API Real-time sentiment and trend data from social media. Internal PDF content,
CRM/ERP logs, user feedback, internal sales data.
2. Data Types
● Free Text Detailed project descriptions, addressed issues, company mission (e.g., "Reduce
downtime by 20%").
● Structured Entities Company name, sector (e.g., manufacturing), competitors, clients,
suppliers.
● Economic Data Market size (in millions ), trends (annual growth), forecasts, entry barriers.
● Qualitative Data Competitive strengths (SWOT analysis), key success factors, user feedback
(e.g., customer satisfaction).
● AI Scores & Analyses Results of Porters Five Forces analysis, Lean recommendations, per
formance projections.
● Conversational Data Chat history with the strategic agent, improvement suggestions (e.g.,
"Optimize storage space").
● Metadata Submission date, user ID, plan version (e.g., v1.2).
Estimated Volume (for MVP Phase)
5.Fonctionnalités attendues
● Analyse des fichiers PDF au niveau automatique : extraction du contexte stratégique
via approche NLP multilingue.
● Générateur de Business Plan : rédaction synthétique en langue naturelle +
exportation au format (PDF).
● Dashboard interactif : visualisation des forces de Porter sous forme de radar,
tableaux, heatmaps.
● Recommandations arborescentes : suggérées par agents IA selon forces en présence.
● Roadmap Produit : préconisation des fonctionnalités à développer avec priorité.
Element Estimated Volume Comment
PDF Files 100-200 documents Provided by internal testers or pilot
partners (e.g., Form Entries
Form Entries 150-300 projects
Manual collection alongside PDF
files (e.g., 200 validated projects).
AI Agent Queries 1,500 executions 35 agents/project OE 300 projects,
managed by local LLM or External API
Saved Versions 900 versions
3 versions/project on average, stored
in IPFS or versioned SQL database.
● Assistant conversationnel : chatbot intelligent intégrant l'ensemble des analyses afin
d’interagir avec l’utilisateur.
● Exportation du rapport imprimable / présentable.
6. Contraintes techniques
- Environnement cible : Web app responsive (desktop/mobile)
- Langages / frameworks privilégiés : Python (FastAPI), Django, React.js, LangChain ou
CrewAI pour les agents, PostgreSQL
- Interopérabilité avec systèmes existants : : Intégration possible avec SharePoint, Google
Drive ou ERP internes (phase 2)
- Sécurité et confidentialité :Accès restreint, chiffrement des données, authentification
firebase authentication
7. Méthodologie proposée
Agile : Kanban
8. Critères de succès / KPI
- Précision / Recall / F-score :
L’équipe souhaite atteindre un F-score supérieur à 0,85. En effet, les modèles utilisés
emploieront un “large” nombre de paramètres et seront entraînés sur des jeux de données
“larges”.
- Réduction de temps / coût :
La réalisation d’un rapport en utilisant la méthodologie des 5 forces de Porter
nécessite de conduire une analyse de marché détaillée. Réunir les ressources et le personnel
nécessaires à cela est un processus pouvant prendre plusieurs semaines, voire plusieurs
mois pour une entreprise de grande envergure. Il en est de même quant à la mobilisation de
personnel au sein de l’entreprise ou au recrutement de sous-traitants (comme une
entreprise spécialisée dans le conseil). La solution proposée doit permettre d’éliminer ces
coûts presque entièrement.
- Satisfaction des utilisateurs finaux :
L’interface doit bénéficier d’une prise en main intuitive et d’une esthétique
minimaliste, permettant aux utilisateurs de focaliser leur attention sur le contenu de leurs
travaux. La transformation d’un ensemble d’idées en un rapport complet doit être un
processus direct, c’est-à-dire sans étapes intermédiaires. L’équipe vise un taux de
satisfaction — l’équivalent de notes issues des opinions d’utilisateurs sur des sources de
téléchargement d’applications ou données à travers des enquêtes d’opinions — d’au moins
90% (en notant que les utilisateurs susmentionnés représentent des professionnels au sein
d’entreprises, le produit étant à visée B2B).
9. Équipe-projet
- Chef de projet : Martin Kirilov-Lilov
- Développeur(s) : Aziz Dridi, Ilef Bennour, Nour Gaboussa, Sarra Mouadeb
- Référent métier : Aya Ben Hmida
10. Livrables attendus
- Rapport de cadrage:
1. Analyse des besoins des différents utilisateurs : entrepreneurs, étudiants, analystes
2. Architecture technique: développement de backend avec Django, et du frontend avec
React.
3. Planning Agile (sprints de 2 semaines, priorisation des features)
- Dataset préparé:
1. 500+ PDFs de business plans ou études de marché labellisés manuellement.
2. Scripts de prétraitement (nettoyage texte, extraction tables/figures).
- Modèle IA entraîné et documenté:
1. Modèle NLP (SpaCy + BERT) pour classifier les 5 forces depuis le texte
2. Module de génération (GPT-like) pour les recommandations stratégiques
3. Fiche technique : précision >85% sur jeu de test, temps d’inférence <5s
- Application / API / Dashboard:
1. Application Web (Frontend)
1.1. Fonctionnalités claires :
1.1.1. Page d’upload (drag & drop de PDF/TXT) avec aperçu du fichier
1.1.2. Bouton "Analyser" déclenchant le traitement IA
1.1.3. Dashboard interactif :
- Visualisation des 5 forces (diagramme radar + scores en %)
-Onglet "Recommandations" (liste priorisée d’améliorations)
-Bouton "Générer le Business Plan" (PDF en 5 blocs)
2. API (Backend):
2.1. Endpoints documentés :
2.1.1. POST /api/upload → Reçoit le PDF, retourne un ID d’analyse
2.2.2. GET /api/results/{id} → Retourne les scores 5P + recommandations
2.2.3. POST /api/generate-pdf → Génère le business plan en PDF
- Documentation technique & utilisateur:
1. Utilisateur :
1.1. Fiche "Lire votre analyse 5P"
1.2. Tutoriel vidéo pour utiliser le dashboard
2. Technique :
2.1. Guide d'installation (variables d'environnement)
2.2. Swagger/OpenAPI pour l’API
    """

    team = {
        "sprint_length_days": 14,
        "sprint_capacity_points": 20,
        "members": [
            {"name": "Sami", "role": "Backend", "skills": ["backend", "sql", "spring"]},
            {"name": "Ali", "role": "Frontend", "skills": ["frontend", "angular", "ui"]},
            {"name": "Mouna", "role": "DevOps", "skills": ["devops", "docker", "ci_cd"]},
            {"name": "Hela", "role": "QA", "skills": ["qa", "testing"]}
        ]
    }

    result = graph.invoke({
        "cahier_de_charge": cahier_de_charge,
        "team": team,
        "validation_attempts": 0,
        "max_validation_attempts": 3
    })

    print(result)
    if result["spec_validation"]["ok"] == False:
        print("validation failed")
        print(result["spec_validation"]["errors"])
    else:
        print("============================================")
        print("Requirements:", result["requirements"])
        print("============================================")
        print("Product Backlog:", result["product_backlog"])
        print("============================================")
        print("Estimated Backlog:", result["estimated_backlog"])
        print("============================================")
        print("Dependencies:", result.get("dependencies", []))
        print("============================================")
        print("Sprints:", result["sprint_backlogs"])
        print("============================================")
        print("Assignments:", result["assignments"])
        print("============================================")
        print("Validation:", result["validation"])
        print("============================================")