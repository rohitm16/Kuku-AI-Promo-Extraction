# Scoring Rubric & Evaluation Protocol

This document details the semantic evaluation criteria, composite scoring formulas, penalty multipliers, and the Cold-Viewer protocol used by the **Kuku AI Promo Extraction System**.

---

## 1. The Cold-Viewer Evaluation Protocol

Mobile microdrama marketing (TikTok, Instagram Reels, YouTube Shorts) relies on acquiring uninitiated viewers scrolling through high-velocity feeds. 

Traditional LLM evaluation prompts often fail because they suffer from the **"Curse of Knowledge"**—the model is given full-season synopses or episode summaries, leading it to assume viewers understand character backstories, motives, and lore.

### Context Isolation
The prompt instructs the model to use only the clip. This is not enforced:
* The prompt contains the clip, the next beat, and two other candidates from the same batch.
* The model is explicitly instructed: *"You are an uninitiated viewer who has NEVER watched this show before. If an uninitiated stranger cannot understand who is fighting or what is at stake from this slice alone, flag `requires_prior_lore: true`."*
* Clips that rely on unexplained family trees, off-screen relationships, or obscure magical/legal concepts are heavily penalized.

---

## 2. Mathematical Scoring Formulation

The final ranking metric is the **Composite Score** ($S_{\text{composite}}$), computed via a weighted base score modulated by binary penalty multipliers:

$$S_{\text{composite}} = \left(0.35 \cdot S_{\text{hook}} + 0.35 \cdot S_{\text{cliff}} + 0.30 \cdot S_{\text{clarity}}\right) \times P_{\text{lookahead}} \times P_{\text{lore}}$$

### Component Weights

| Component | Weight | Metric Range | Focus & Marketing Rationale |
| :--- | :---: | :---: | :--- |
| **Hook Score ($S_{\text{hook}}$)** | **35%** | 1.0 – 10.0 | **First 3-Second Scroll-Stopping Power:** Evaluates kinetic visual shock, audio confrontation, or audiovisual sync that halts feed scrolling. |
| **Cliffhanger Score ($S_{\text{cliff}}$)** | **35%** | 1.0 – 10.0 | **Cut-Point Unresolved Tension:** Evaluates the existential, physical, or social jeopardy left hanging when the clip cuts to black. |
| **Standalone Clarity ($S_{\text{clarity}}$)** | **30%** | 1.0 – 10.0 | **Self-Sufficiency for Uninitiated Viewers:** Evaluates whether the protagonist, antagonist, and immediate stakes are discernible from this isolated slice. |

---

## 3. Penalty Multipliers & Quality Safeguards

To prevent the pipeline from rewarding deceptive, deflated, or confusing clips, two deterministic penalty multipliers are applied:

### 1. Next-Beat Lookahead Penalty ($P_{\text{lookahead}} = 0.70$)
* **The Artificial Cliffhanger Trap:** A naive algorithm might cut a scene at $[00:30.00]$ immediately after a character asks: *"Whose carriage is arriving outside?"* To an isolated observer, this appears to be a high-tension question. However, if the immediate next beat at $[00:30.50]$ reveals: *"It's just the grocer delivering flour"*, the cliffhanger is artificial and immediately deflated.
* **The Lookahead Probe:** Along with the candidate slice, the evaluator is provided with the immediate continuation $[T_{\text{end}}, T_{\text{next\_beat}}]$. If the lookahead text immediately answers or trivializes the dramatic question, the model flags `is_immediately_resolved: true`, triggering a **30% score penalty** ($P_{\text{lookahead}} = 0.70$).

### 2. Prior Lore Dependency Penalty ($P_{\text{lore}} = 0.50$)
* **The Continuity Trap:** Many high-drama scenes involve intricate third-party noble lineage disputes or obscure inheritance deeds. While intense for existing fans, an uninitiated user on TikTok will find them bewildering.
* **The Lore Penalty:** If the conflict cannot be understood without knowledge from previous episodes, the model flags `requires_prior_lore: true`, triggering a **50% score penalty** ($P_{\text{lore}} = 0.50$).

---

## 4. Hook & Cliffhanger Classification Schema

The evaluator classifies each candidate into specific narrative archetypes:

### Hook Types
1. **`visual`**: Kinetic physical action, slaps, drawn weapons, abrupt door crashes, high visual contrast (e.g., bride in white standing alone at the altar).
2. **`audio_verbal`**: Shocking dialogue lines, public insults, lethal threats, wedding objections.
3. **`audiovisual_sync`**: Synchronized sound and image (e.g., a gavel striking the court sound block simultaneously with a guilty verdict pronouncement).

### Cliffhanger Types
1. **`scene_terminal`**: Natural narrative cliffhanger coinciding with a scene cutoff, door slam, or cut to black.
2. **`intra_scene`**: A mid-scene escalation cut immediately before an impending violent or emotional clash.

---

## 5. Pydantic Structured Output Schema

The evaluator response is strictly validated against the following Pydantic model (`SinglePromoEvaluation`):

```python
class SinglePromoEvaluation(BaseModel):
    candidate_id: str
    hook_score: float = Field(ge=1.0, le=10.0)
    hook_type: str = Field(description="visual | audio_verbal | audiovisual_sync")
    hook_description: str
    cliffhanger_score: float = Field(ge=1.0, le=10.0)
    cliffhanger_type: str = Field(description="scene_terminal | intra_scene")
    cliffhanger_question: str
    is_immediately_resolved: bool
    standalone_clarity_score: float = Field(ge=1.0, le=10.0)
    identified_conflict: str
    requires_prior_lore: bool
    editorial_rationale: str
```

If the upstream LLM returns invalid JSON or violates field bounds, the client adapter catches the error and retries with self-correcting schema guidance (up to 2 retries).

---

## 6. Diversity-Aware Ranking & Deduplication Protocol

After composite scoring, the candidates are processed through a dual ranking regime:

### 1. Intra-Episode Temporal Deduplication ($\text{IoU} \le 0.50$)
Multiple shortlisted candidate windows within the same episode may cover overlapping portions of a single dramatic climax. 
We calculate temporal Intersection-over-Union:
$$\text{IoU}(A, B) = \frac{\max(0, \min(A_{\text{end}}, B_{\text{end}}) - \max(A_{\text{start}}, B_{\text{start}}))}{\max(A_{\text{end}}, B_{\text{end}}) - \min(A_{\text{start}}, B_{\text{start}})}$$
When $\text{IoU}(A, B) > 0.50$, only the higher-scoring promo is preserved. This reduces 463 evaluated candidates to **282 distinct, non-redundant promotional clips**.

### 2. Global Catalog Diversity Guardrail
To ensure a broad ad creative catalog and prevent hyper-dramatic episodes (e.g., Episode 15 or 36) from monopolizing user acquisition ad spend:
* **Top 10 Guardrail:** At most 1 promo per episode is permitted in the global Top 10.
* **Top 25 Guardrail:** At most 2 promos per episode are permitted in the global Top 25.

