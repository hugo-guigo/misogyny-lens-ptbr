"""Second opinion from Claude: label, confidence and a short explanation in Portuguese.

Used two ways:
- offline, by eval_llm.py, to score the 333-sentence test set;
- online, by POST /api/llm-explain, only when ANTHROPIC_API_KEY is configured.

The definition follows the project's taxonomy (7 classes: discredit, stereotyping,
sexual harassment, threats of violence, dominance, victim blaming, sexual objectification)
and its methodological note: a word alone is not misogyny, the occurrence in context is.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import anthropic

MODEL = os.environ.get("LLM_MODEL", "claude-opus-5")

SYSTEM = """Você é um anotador especialista em misoginia digital em português brasileiro.
Sua tarefa: dizer se um texto curto de rede social tem traços de misoginia.

Definição (Manne, 2017): misoginia é a dimensão punitiva e de controle contra mulheres.
Considere traços de misoginia quando o texto, no contexto, faz alguma destas coisas
contra mulheres ou uma mulher por ser mulher:
1. Descrédito: ofende ou desqualifica (ex.: xingamentos de gênero como "vaca", "piranha").
2. Estereotipização: reforça papéis ou incapacidades femininas ("lugar de mulher é...").
3. Assédio sexual: comentários sexuais indesejados dirigidos a alguém.
4. Ameaças de violência contra mulheres.
5. Dominação: afirma controle ou superioridade masculina sobre mulheres.
6. Culpabilização da vítima: culpa a mulher pela violência que sofreu.
7. Objetificação sexual: reduz a mulher a corpo ou objeto sexual.

Não marque como misoginia só porque o texto menciona mulheres, feminismo ou gênero,
nem insultos genéricos sem alvo feminino, nem críticas políticas a uma mulher que não
usam o gênero dela como arma. Palavras isoladas não bastam: julgue o uso no contexto.
Responda apenas com o JSON pedido."""

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "integer", "enum": [0, 1],
                  "description": "1 se o texto tem traços de misoginia, 0 caso contrário"},
        "confidence": {"type": "number", "description": "confiança entre 0 e 1"},
        "category": {"type": "string",
                     "enum": ["nenhuma", "descredito", "estereotipizacao", "assedio_sexual",
                              "ameacas", "dominacao", "culpabilizacao_da_vitima", "objetificacao_sexual"]},
        "explanation": {"type": "string", "description": "uma ou duas frases em português"},
    },
    "required": ["label", "confidence", "category", "explanation"],
    "additionalProperties": False,
}


@dataclass
class Judgment:
    label: int
    confidence: float
    category: str
    explanation: str
    model: str
    refused: bool = False


def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(max_retries=4)


def judge(text: str, client: anthropic.Anthropic | None = None) -> Judgment:
    client = client or get_client()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Texto:\n<texto>\n{text}\n</texto>"}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        # Server-side fallback: if the model declines (a safety classifier can fire on
        # hostile text), the API retries on a fallback model within the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return Judgment(label=0, confidence=0.0, category="nenhuma",
                        explanation="O modelo recusou analisar este texto.", model=response.model, refused=True)
    raw = next(b.text for b in response.content if b.type == "text")
    data = json.loads(raw)
    return Judgment(label=int(data["label"]), confidence=float(data["confidence"]),
                    category=data["category"], explanation=data["explanation"], model=response.model)
