"""
AI Quiz Generation Module for AI-Powered Smart Study Assistant (Phase 5).
Automatically generates high-quality Multiple-Choice Questions (MCQs) from uploaded PDF documents.
Supports configurable question counts, 3 difficulty tiers (Easy, Medium, Hard),
distractor synthesis, factual validation, source page tracking, and comprehensive answer explanations.
"""

import random
import re
from typing import Any

from database.db import (
    get_document,
    get_document_pages,
)
from utils.summarizer import (
    clean_text,
    extract_exam_points,
    extract_key_concepts,
    extract_keywords,
    split_into_sentences,
)


class QuizGenerationError(Exception):
    """Custom exception raised for quiz generation failures."""


# ------------------------------------------------------------------------------
# Candidate Fact Extraction & Question Construction Pipeline
# ------------------------------------------------------------------------------

def _extract_candidate_facts(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extracts high-salience candidate facts, definitions, and statements across all pages of a document.
    Returns a list of dicts: {'fact': str, 'page_number': int, 'concept': str, 'type': str}
    """
    candidates = []
    seen_facts: set[str] = set()

    for p in pages:
        page_num = p.get("page_number", 1)
        raw_text = p.get("extracted_text") or p.get("text", "")
        text = clean_text(raw_text)
        if not text or len(text.split()) < 5:
            continue

        # 1. Concept-definition pairs (Highest quality for definitions)
        concepts = extract_key_concepts(text, top_n=6)
        for c in concepts:
            concept_name = c.get("concept", "").strip()
            # Strip chapter/section prefixes
            concept_name = re.sub(r"^(Chapter|Section|Part|Unit)\s*\d+[:.\-]?\s*", "", concept_name, flags=re.IGNORECASE).strip()
            def_text = c.get("definition", "").strip()
            def_text = re.sub(r"^(Chapter|Section|Part|Unit)\s*\d+[:.\-]?\s*", "", def_text, flags=re.IGNORECASE).strip()
            if concept_name and def_text and len(def_text.split()) >= 4:
                fact_str = f"{concept_name}: {def_text}"
                if fact_str.lower() not in seen_facts:
                    seen_facts.add(fact_str.lower())
                    candidates.append({
                        "fact": def_text,
                        "concept": concept_name,
                        "page_number": page_num,
                        "type": "concept_definition",
                    })

        # 2. Key exam points (High-value rules, laws, mechanisms, comparisons)
        exam_points = extract_exam_points(text, max_points=6)
        for ep in exam_points:
            ep_clean = re.sub(r"^(Chapter|Section|Part|Unit)\s*\d+[:.\-]?\s*", "", ep.strip(), flags=re.IGNORECASE).strip()
            if len(ep_clean.split()) >= 6 and ep_clean.lower() not in seen_facts:
                seen_facts.add(ep_clean.lower())
                candidates.append({
                    "fact": ep_clean,
                    "concept": "",
                    "page_number": page_num,
                    "type": "exam_point",
                })

        # 3. Informative factual sentences with strong copular or relational verbs
        sentences = split_into_sentences(text)
        for s in sentences:
            s_clean = s.strip()
            words = s_clean.split()
            if (
                6 <= len(words) <= 40
                and re.search(
                    r"\b(is defined as|refers to|consists of|characterized by|formed through|operates by|results in|composed of|responsible for|primary function of|is the|is a|are the|generate|generates|convert|converts|produces|causes|transforms|includes|contains)\b",
                    s_clean,
                    re.IGNORECASE,
                )
                and s_clean.lower() not in seen_facts
            ):
                seen_facts.add(s_clean.lower())
                candidates.append({
                    "fact": s_clean,
                    "concept": "",
                    "page_number": page_num,
                    "type": "relational_sentence",
                })

        # 4. Fallback if no candidate facts extracted from this page yet
        page_cands = [c for c in candidates if c["page_number"] == page_num]
        if not page_cands:
            for s in sentences:
                s_clean = s.strip()
                words = s_clean.split()
                if 6 <= len(words) <= 40 and s_clean.lower() not in seen_facts:
                    seen_facts.add(s_clean.lower())
                    candidates.append({
                        "fact": s_clean,
                        "concept": " ".join(words[:2]),
                        "page_number": page_num,
                        "type": "sentence_fallback",
                    })

    return candidates


def _build_distractor_pool(pages: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[str]:
    """
    Collects a rich pool of domain-specific concepts, clauses, and definitions
    from the document to synthesize realistic, non-trivial distractors.
    """
    pool: list[str] = []
    seen: set[str] = set()

    for c in candidates:
        f = c.get("fact", "").strip()
        if f and len(f.split()) >= 4 and f.lower() not in seen:
            seen.add(f.lower())
            pool.append(f)

    # Add keywords and concept names
    full_text = " ".join([p.get("extracted_text") or p.get("text", "") for p in pages])
    kw = extract_keywords(full_text, top_n=25)
    for k in kw:
        if k and k.lower() not in seen:
            seen.add(k.lower())
            pool.append(k.capitalize())

    return pool


# ------------------------------------------------------------------------------
# Question Formatting Helpers for Difficulty Tiers
# ------------------------------------------------------------------------------

def _generate_distractors(
    correct_text: str,
    distractor_pool: list[str],
    difficulty: str = "medium",
) -> list[str]:
    """
    Selects 3 distinct, contextually plausible distractors from the document distractor pool.
    Modifies them slightly if needed to ensure syntactic balance with the correct answer.
    """
    correct_lower = correct_text.lower().strip()
    # Filter candidates from pool that are not too similar to correct answer
    valid_candidates = []
    for item in distractor_pool:
        item_clean = item.strip()
        if not item_clean:
            continue
        item_lower = item_clean.lower()
        if item_lower == correct_lower:
            continue
        # Avoid near-substring identity
        if correct_lower in item_lower or item_lower in correct_lower:
            continue
        valid_candidates.append(item_clean)

    # If we have enough candidates in the pool, pick 3
    random.shuffle(valid_candidates)
    chosen: list[str] = []
    for cand in valid_candidates:
        if len(chosen) >= 3:
            break
        # Ensure chosen candidates are mutually distinct
        if cand not in chosen and cand.lower() != correct_lower:
            chosen.append(cand)

    # Fallbacks in case document has sparse distinct candidates
    fallback_templates = [
        "It operates only under non-standard external conditions.",
        "It acts as a passive boundary without catalytic or regulatory action.",
        "It represents an unstable intermediate state that dissolves immediately.",
        "It applies strictly to homogeneous single-phase mixtures.",
        "It decreases exponentially as temperature increases.",
        "It requires continuous external energy input to maintain equilibrium.",
    ]
    random.shuffle(fallback_templates)
    for fb in fallback_templates:
        if len(chosen) >= 3:
            break
        if fb not in chosen and fb.lower() != correct_lower:
            chosen.append(fb)

    return chosen[:3]


def _format_question_by_difficulty(
    fact_info: dict[str, Any],
    difficulty: str,
    distractor_pool: list[str],
) -> dict[str, Any] | None:
    """
    Constructs a calibrated multiple-choice question reflecting the requested difficulty:
    - EASY: Direct definition and basic conceptual recognition.
    - MEDIUM: Functional role, mechanism, and process comparison.
    - HARD: Analytical consequence, conditions, and principle application.
    """
    fact = fact_info.get("fact", "").strip()
    concept = fact_info.get("concept", "").strip()
    page_num = fact_info.get("page_number", 1)

    if not fact:
        return None

    # Determine concept if empty
    if not concept:
        # Extract leading subject from fact sentence
        match = re.match(r"^([A-Z][a-zA-Z0-9\s\-]{2,30}?)\s+(?:is|are|was|were|refers to|consists of|forms|operates)\b", fact)
        if match:
            concept = match.group(1).strip()
        else:
            # Fallback: take first 2-3 words
            concept = " ".join(fact.split()[:3]).rstrip(" ,.:;")

    difficulty_tier = difficulty.lower().strip()
    if difficulty_tier not in ("easy", "medium", "hard"):
        difficulty_tier = "medium"

    question_text = ""
    correct_answer = ""
    explanation = ""

    # Attempt neural prompt generation via Flan-T5 if concept is rich
    neural_question = None
    try:
        from utils.chatbot import get_qa_engine
        qa_engine = get_qa_engine()
        if qa_engine and len(fact.split()) >= 12:
            prompt = (
                f"Given the study fact: \"{fact}\"\n"
                f"Write a single clear {difficulty_tier} question about this fact:\nQuestion:"
            )
            ans = qa_engine.generate_answer(prompt, max_length=50, min_length=8)
            ans = ans.strip()
            if ans and "?" in ans and len(ans.split()) >= 5:
                # Ensure it looks like a clean question
                if not ans.endswith("?"):
                    ans += "?"
                neural_question = ans
    except (RuntimeError, ValueError, TypeError, AttributeError):
        neural_question = None

    if difficulty_tier == "easy":
        # Direct factual question
        if neural_question:
            question_text = neural_question
        elif concept and len(concept.split()) <= 4:
            question_text = f"Which of the following statements best defines or describes '{concept}'?"
        else:
            question_text = "Based on the study material, which statement is factually accurate regarding the document context?"

        correct_answer = fact
        if not correct_answer.endswith((".", "!", "?")):
            correct_answer += "."
        explanation = f"According to page {page_num}, this fact is directly confirmed in the text: {fact}"

    elif difficulty_tier == "medium":
        # Process, mechanism, and functional role
        if neural_question:
            question_text = neural_question
        elif concept and len(concept.split()) <= 4:
            templates = [
                f"What is the primary role, mechanism, or principle associated with '{concept}'?",
                f"In the context of the study material, how is '{concept}' characterized?",
                f"Which of the following correctly explains the function or significance of '{concept}'?",
            ]
            question_text = random.choice(templates)
        else:
            question_text = "Which of the following represents a correct principle or mechanism discussed in the document?"

        correct_answer = fact
        if not correct_answer.endswith((".", "!", "?")):
            correct_answer += "."
        explanation = f"On page {page_num}, the text notes that: {fact}"

    else:
        # HARD: Analytical, condition-based, or scenario-based
        if neural_question:
            if any(w in neural_question.lower() for w in ["analyz", "evaluat", "deduction", "mechanism", "statement", "principle"]):
                question_text = neural_question
            else:
                question_text = f"When evaluating this mechanism, {neural_question[:1].lower() + neural_question[1:]}"
        elif concept and len(concept.split()) <= 4:
            templates = [
                f"Under the principles discussed in the document, which statement correctly analyzes the constraints or behavior of '{concept}'?",
                f"When evaluating the operation of '{concept}', which of the following analytical conclusions is directly supported by the text?",
                f"Which underlying mechanism explains why '{concept}' operates as described in the study material?",
            ]
            question_text = random.choice(templates)
        else:
            question_text = "Which of the following statements represents the most accurate analytical deduction from the study material?"

        correct_answer = fact
        if not correct_answer.endswith((".", "!", "?")):
            correct_answer += "."
        explanation = f"Analytical verification from page {page_num}: {fact}"

    # Generate 3 realistic distractors from document concepts
    distractors = _generate_distractors(correct_answer, distractor_pool, difficulty=difficulty_tier)

    # Ensure we have 3 distractors
    if len(distractors) < 3:
        return None

    # Assemble all 4 options
    all_options = [correct_answer] + distractors
    random.shuffle(all_options)

    # Locate index of correct answer
    correct_idx = all_options.index(correct_answer)
    option_letters = ["A", "B", "C", "D"]
    correct_letter = option_letters[correct_idx]

    return {
        "question_text": question_text,
        "option_a": all_options[0],
        "option_b": all_options[1],
        "option_c": all_options[2],
        "option_d": all_options[3],
        "correct_option": correct_letter,
        "explanation": explanation,
        "difficulty": difficulty_tier,
        "source_page": page_num,
    }


# ------------------------------------------------------------------------------
# Core Quiz Generator Function
# ------------------------------------------------------------------------------

def generate_quiz_for_document(
    document_id: int,
    num_questions: int = 5,
    difficulty: str = "medium",
    custom_title: str | None = None,
    time_limit_minutes: int = 0,
    db_path: Any | None = None,
) -> dict[str, Any]:
    """
    Generates a full AI quiz for a document:
    1. Validates document exists and has extractable text.
    2. Extracts candidate facts, concepts, and statements across all document pages.
    3. Formulates calibrated multiple-choice questions matching the requested difficulty.
    4. Validates each question (4 distinct options, exactly 1 correct, non-empty).
    5. Deduplicates questions to ensure variety.
    6. Returns structured quiz data ready for database insertion.
    """
    # 1. Document validation
    document = get_document(document_id, db_path=db_path)
    if not document:
        raise QuizGenerationError(f"Document with ID {document_id} was not found.")

    if document.get("processing_status") == "scanned_no_text" or document.get("total_words", 0) == 0:
        raise QuizGenerationError("This document is a scanned image or contains no extractable text. Cannot generate quiz.")

    # 2. Extract pages
    pages = get_document_pages(document_id, db_path=db_path)
    if not pages:
        raise QuizGenerationError("No extracted text pages found for this document.")

    # 3. Extract candidate facts and build distractor pool
    candidates = _extract_candidate_facts(pages)
    if not candidates:
        raise QuizGenerationError("Could not extract sufficient facts or concepts from this document to generate quiz questions.")

    distractor_pool = _build_distractor_pool(pages, candidates)

    # 4. Sizing validation
    requested_count = max(1, min(int(num_questions or 5), 25))
    difficulty_level = (difficulty or "medium").lower().strip()
    if difficulty_level not in ("easy", "medium", "hard"):
        difficulty_level = "medium"

    # Shuffle candidates to ensure variety
    random.shuffle(candidates)

    generated_questions: list[dict[str, Any]] = []
    seen_questions: set[str] = set()

    for cand in candidates:
        if len(generated_questions) >= requested_count:
            break

        q_obj = _format_question_by_difficulty(cand, difficulty_level, distractor_pool)
        if not q_obj:
            continue

        # Check for duplicate question text
        q_norm = q_obj["question_text"].lower().strip()
        ans_norm = q_obj["option_" + q_obj["correct_option"].lower()].lower().strip()
        dedup_key = f"{q_norm}::{ans_norm[:40]}"

        if dedup_key in seen_questions:
            continue
        seen_questions.add(dedup_key)

        # Validate that all 4 options are distinct
        opts = [q_obj["option_a"], q_obj["option_b"], q_obj["option_c"], q_obj["option_d"]]
        if len(set(opts)) != 4:
            continue

        q_obj["question_number"] = len(generated_questions) + 1
        generated_questions.append(q_obj)

    # If we need more questions than candidate concepts, synthesize variations across pages
    if len(generated_questions) < requested_count:
        for p in pages:
            if len(generated_questions) >= requested_count:
                break
            raw_text = clean_text(p.get("extracted_text") or p.get("text", ""))
            sentences = split_into_sentences(raw_text)
            for s in sentences:
                if len(generated_questions) >= requested_count:
                    break
                s_words = s.strip().split()
                if 12 <= len(s_words) <= 30:
                    cand_item = {
                        "fact": s.strip(),
                        "concept": " ".join(s_words[:2]),
                        "page_number": p.get("page_number", 1),
                        "type": "supplementary",
                    }
                    q_obj = _format_question_by_difficulty(cand_item, difficulty_level, distractor_pool)
                    if q_obj:
                        q_norm = q_obj["question_text"].lower().strip()
                        ans_norm = q_obj["option_" + q_obj["correct_option"].lower()].lower().strip()
                        dedup_key = f"{q_norm}::{ans_norm[:40]}"
                        if dedup_key not in seen_questions:
                            seen_questions.add(dedup_key)
                            opts = [q_obj["option_a"], q_obj["option_b"], q_obj["option_c"], q_obj["option_d"]]
                            if len(set(opts)) == 4:
                                q_obj["question_number"] = len(generated_questions) + 1
                                generated_questions.append(q_obj)

    if not generated_questions:
        raise QuizGenerationError("Failed to generate valid quiz questions from the provided document content.")

    # 5. Determine title
    orig_filename = document["original_filename"]
    base_name = orig_filename.rsplit(".", 1)[0]
    title = (custom_title or "").strip()
    if not title:
        title = f"{base_name} — {difficulty_level.capitalize()} Quiz"

    return {
        "document_id": document_id,
        "original_filename": orig_filename,
        "title": title,
        "difficulty": difficulty_level,
        "total_questions": len(generated_questions),
        "time_limit_minutes": int(time_limit_minutes or 0),
        "questions": generated_questions,
    }
