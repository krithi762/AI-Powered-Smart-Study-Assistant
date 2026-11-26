"""
AI Summarization module for AI-Powered Smart Study Assistant.
Provides text preprocessing, token-safe chunking, multi-level AI summarization (Quick, Standard, Detailed),
chapter/page-wise summaries, concept and definition extraction, TF-IDF keyword extraction,
and examination preparation point extraction.
"""

import os
import re
import threading
import unicodedata
from typing import Any, Optional

# Silence HuggingFace Hub Windows symlinks warning
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Lazy import flag for heavyweight ML dependencies
_torch_available = False
_transformers_available = False
_sklearn_available = False

try:
    import torch
    _torch_available = True
except ImportError:
    torch = None

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    _transformers_available = True
except ImportError:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _sklearn_available = True
except ImportError:
    TfidfVectorizer = None


class SummarizationError(Exception):
    """Custom exception raised for unrecoverable summarization errors."""


# ------------------------------------------------------------------------------
# 1. Text Preprocessing & Cleaning (Task 1 & Task 2)
# ------------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Cleans and normalizes extracted PDF text:
    - Normalizes unicode characters (NFKC)
    - Fixes hyphenated words broken across line wraps (e.g., 'trans- \\nformer' -> 'transformer')
    - Removes standalone page markers (e.g., '--- Page 1 ---', 'Page 1 of 5')
    - Removes control characters and excess whitespace
    - Preserves paragraph and sentence structures
    """
    if not text:
        return ""

    # Normalize unicode representations
    cleaned = unicodedata.normalize("NFKC", text)

    # Standardize line breaks
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # Fix hyphens split across line breaks: 'exam-\\nple' -> 'example'
    cleaned = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", cleaned)

    # Remove artificial page markers inserted during extraction
    cleaned = re.sub(r"---+\s*Page\s*\d+\s*---+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*Page\s+\d+\s+(?:of\s+\d+)?\s*$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE)

    # Collapse multiple inline whitespace to single space (preserve newlines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Collapse 3 or more consecutive newlines into 2 (paragraphs)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into coherent sentences using punctuation and common abbreviation heuristics.
    Handles common abbreviations (e.g., Dr., Prof., Fig., e.g., i.e., vs., etc.) and decimal numbers.
    Preserves original casing of abbreviations.
    """
    if not text or not text.strip():
        return []

    # Protect known abbreviations with exact matched casing
    abbreviations = [
        "e.g.", "i.e.", "etc.", "vs.", "al.", "approx.",
        "dr.", "mr.", "mrs.", "ms.", "prof.", "fig.", "eq.",
    ]
    abbr_map: dict[str, str] = {}

    def _replace_abbr(match: re.Match) -> str:
        key = f"__ABBR_{len(abbr_map)}__"
        abbr_map[key] = match.group(0)
        return key

    protected = text
    for abbr in abbreviations:
        pattern = re.compile(r"(?:\b|\s)" + re.escape(abbr), re.IGNORECASE)
        protected = pattern.sub(lambda m: " " + _replace_abbr(m) if m.group(0).startswith(" ") else _replace_abbr(m), protected)

    # Split on sentence boundaries: period, question mark, or exclamation followed by space/newline
    raw_sentences = re.split(r"(?<=[.?!])\s+", protected)

    restored_sentences: list[str] = []
    for s in raw_sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        # Restore abbreviations with original case
        for key, orig_val in abbr_map.items():
            s_clean = s_clean.replace(key, orig_val.strip())
        restored_sentences.append(s_clean)

    return restored_sentences



def chunk_text(
    text: str,
    max_chunk_words: int = 350,
    overlap_words: int = 35,
    max_chunk_tokens: int = 700,
    tokenizer: Any | None = None,
) -> list[str]:
    """
    Splits long documents into manageable chunks without exceeding model token limits (1024).
    Preserves paragraph and sentence boundaries wherever possible.
    Splits unusually long sentences on clause or word boundaries.
    Includes a sliding overlap between chunks to maintain narrative context across boundaries.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    words = cleaned.split()
    if len(words) <= max_chunk_words:
        return [cleaned]

    # Helper to check sentence size (tokens or token estimation)
    def count_tokens_or_words(s: str) -> int:
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            try:
                return len(tokenizer.encode(s, add_special_tokens=False))
            except Exception:
                pass
        return int(len(s.split()) * 1.35)

    limit_count = max_chunk_tokens if tokenizer is not None else int(max_chunk_words * 1.35)
    overlap_limit = int(overlap_words * 1.35)

    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_count = 0

    for paragraph in paragraphs:
        p_sentences = split_into_sentences(paragraph)
        if not p_sentences:
            p_sentences = [paragraph]

        # Break any giant sentence that exceeds limit_count
        normalized_sentences: list[str] = []
        for s in p_sentences:
            s_count = count_tokens_or_words(s)
            if s_count > limit_count:
                sub_parts = re.split(r"(?<=[;:,])\s+", s)
                sub_accum: list[str] = []
                sub_accum_count = 0
                for part in sub_parts:
                    p_cnt = count_tokens_or_words(part)
                    if sub_accum_count + p_cnt > limit_count and sub_accum:
                        normalized_sentences.append(" ".join(sub_accum))
                        sub_accum = [part]
                        sub_accum_count = p_cnt
                    else:
                        sub_accum.append(part)
                        sub_accum_count += p_cnt
                if sub_accum:
                    normalized_sentences.append(" ".join(sub_accum))
            else:
                normalized_sentences.append(s)

        for sentence in normalized_sentences:
            s_cnt = count_tokens_or_words(sentence)
            if current_count + s_cnt > limit_count and current_sentences:
                chunk_str = " ".join(current_sentences).strip()
                if chunk_str:
                    chunks.append(chunk_str)

                # Keep overlap sentences from tail
                overlap_acc: list[str] = []
                overlap_cnt = 0
                for sent in reversed(current_sentences):
                    sent_cnt = count_tokens_or_words(sent)
                    if overlap_cnt + sent_cnt <= overlap_limit:
                        overlap_acc.insert(0, sent)
                        overlap_cnt += sent_cnt
                    else:
                        break

                current_sentences = overlap_acc
                current_count = overlap_cnt

            current_sentences.append(sentence)
            current_count += s_cnt

    if current_sentences:
        final_chunk = " ".join(current_sentences).strip()
        if final_chunk and (not chunks or final_chunk != chunks[-1]):
            chunks.append(final_chunk)

    return chunks


# ------------------------------------------------------------------------------
# 2. AI Summarization Engine (Singleton Pattern - Task 3 & Task 7)
# ------------------------------------------------------------------------------

class SummarizerEngine:
    """
    Singleton AI Summarization engine.
    Loads the pre-trained HuggingFace Seq2Seq model once and caches it in memory.
    Thread-safe and optimized for CPU inference with torch.inference_mode().
    """

    _instance: Optional["SummarizerEngine"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: str | None = None, cache_dir: str | None = None):
        self.model_name = model_name or os.getenv("SUMMARIZATION_MODEL", "sshleifer/distilbart-cnn-12-6")
        if cache_dir:
            self.cache_dir = str(cache_dir)
        else:
            try:
                from config import Config
                self.cache_dir = str(Config.HF_CACHE_DIR)
            except Exception:
                self.cache_dir = os.getenv("HF_HOME", os.path.join(os.getcwd(), "models", "cache"))
        self.tokenizer = None
        self.model = None
        self._is_loaded = False
        self._load_lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_name: str | None = None, cache_dir: str | None = None) -> "SummarizerEngine":
        """Thread-safe accessor for the singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(model_name, cache_dir=cache_dir)
            elif model_name and cls._instance.model_name != model_name:
                # If a different model name is explicitly requested, reinitialize
                cls._instance = cls(model_name, cache_dir=cache_dir)
            return cls._instance

    def load_model(self) -> None:
        """
        Loads the HuggingFace model and tokenizer into memory if not already loaded.
        Enables evaluation mode and sets CPU threading.
        """
        if self._is_loaded and self.model is not None and self.tokenizer is not None:
            return

        with self._load_lock:
            if self._is_loaded and self.model is not None:
                return

            # Check for mock testing mode
            if self.model_name.lower() in ("mock", "test", "testing"):
                self._is_loaded = True
                return

            if not _transformers_available or not _torch_available or torch is None or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
                raise SummarizationError(
                    "Required libraries 'torch' and 'transformers' are not available in the current environment."
                )

            try:
                # Optimize CPU multi-threading
                if hasattr(torch, "set_num_threads"):
                    optimal_threads = min(4, max(1, (os.cpu_count() or 2) // 2))
                    torch.set_num_threads(optimal_threads)

                if self.cache_dir:
                    os.environ.setdefault("HF_HOME", self.cache_dir)

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    low_cpu_mem_usage=True,
                )
                self.model.eval()

                # Suppress forced_bos_token_id warnings
                if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
                    self.model.generation_config.forced_bos_token_id = 0
                if hasattr(self.model.config, "forced_bos_token_id"):
                    self.model.config.forced_bos_token_id = 0

                self._is_loaded = True

            except Exception as e:
                self._is_loaded = False
                self.tokenizer = None
                self.model = None
                raise SummarizationError(
                    f"Failed to load summarization model '{self.model_name}': {e!s}"
                ) from e

    def is_loaded(self) -> bool:
        """Returns True if the underlying model is loaded in memory."""
        return self._is_loaded

    def unload(self) -> None:
        """Releases the model and tokenizer from memory to free system resources."""
        with self._load_lock:
            self.model = None
            self.tokenizer = None
            self._is_loaded = False
            if _torch_available and torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_summary_for_chunk(
        self,
        text: str,
        max_length: int = 140,
        min_length: int = 40,
        num_beams: int = 2,
    ) -> str:
        """
        Generates an abstractive summary for a single text chunk using the Seq2Seq model.
        """
        cleaned = clean_text(text)
        if not cleaned:
            return ""

        words = cleaned.split()
        # If the input text is already shorter than min_length, return as is
        if len(words) <= min_length:
            return cleaned

        # Mock testing mode fallback
        if self.model_name.lower() in ("mock", "test", "testing"):
            sentences = split_into_sentences(cleaned)
            return " ".join(sentences[:3]) if sentences else cleaned[:200]

        self.load_model()
        if self.tokenizer is None or self.model is None or torch is None:
            raise SummarizationError("Summarizer model, tokenizer, or torch runtime is not available.")

        try:
            inputs = self.tokenizer(
                cleaned,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding=False,
            )

            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")

            # Adjust max_length if input is smaller than default
            input_length = input_ids.shape[1]
            effective_max_length = min(max_length, max(min_length + 10, int(input_length * 0.85)))
            effective_min_length = min(min_length, max(10, int(effective_max_length * 0.35)))

            with torch.inference_mode():
                summary_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=effective_max_length,
                    min_length=effective_min_length,
                    num_beams=num_beams,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    length_penalty=1.0,
                )

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            # Clean up spacing around punctuation
            summary = re.sub(r"\s+([.,;:!?])", r"\1", summary)
            return summary

        except Exception as e:
            raise SummarizationError(f"Error during AI model inference: {e!s}") from e


# ------------------------------------------------------------------------------
# 3. Concept, Keyword, and Exam Point Extraction (Task 4)
# ------------------------------------------------------------------------------

def extract_keywords(text: str, top_n: int = 15) -> list[str]:
    """
    Extracts high-significance keywords and key phrases using TF-IDF vectorization.
    Explicitly distinguishes statistical keywords from AI-generated text.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # Fallback frequency extractor if scikit-learn is unavailable
    if not _sklearn_available or TfidfVectorizer is None:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", cleaned.lower())
        stop_words = {
            "the", "and", "that", "have", "for", "not", "with", "you", "this", "but",
            "his", "from", "they", "say", "her", "she", "will", "one", "all", "would",
            "there", "their", "what", "out", "about", "who", "get", "which", "when",
            "make", "can", "like", "time", "just", "him", "know", "take", "people",
            "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think",
            "also", "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these", "give",
            "day", "most", "us", "are", "were", "been", "has", "had", "page", "text",
        }
        filtered = [w for w in words if w not in stop_words]
        counts: dict[str, int] = {}
        for w in filtered:
            counts[w] = counts.get(w, 0) + 1
        sorted_words = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [w[0].title() for w in sorted_words[:top_n]]

    try:
        # Use TF-IDF with unigrams and bigrams
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        )
        # Split text into sentence-level documents to compute document frequency
        docs = split_into_sentences(cleaned)
        if len(docs) < 2:
            docs = [cleaned]

        tfidf_matrix = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()

        # Calculate average TF-IDF score across sentences
        scores = tfidf_matrix.sum(axis=0).A1
        scored_words = list(zip(feature_names, scores))
        scored_words.sort(key=lambda x: x[1], reverse=True)

        # Filter out generic words that might bypass stop_words
        generic_noise = {"page", "section", "figure", "table", "chapter", "example", "using", "shown"}
        filtered_keywords = []
        for word, _ in scored_words:
            if word.lower() not in generic_noise and not word.isdigit():
                # Title-case for display
                filtered_keywords.append(word.title())
            if len(filtered_keywords) >= top_n:
                break

        return filtered_keywords

    except (ValueError, TypeError, AttributeError, RuntimeError):
        # Fallback to simple regex-based word frequency on error
        words = re.findall(r"\b[a-zA-Z]{4,}\b", cleaned.title())
        counts: dict[str, int] = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        return [w[0] for w in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]]


def extract_key_concepts(text: str, top_n: int = 8) -> list[dict[str, str]]:
    """
    Extracts important concepts and formal definitions from the study material using
    linguistic pattern matching.
    Patterns detect structures like:
    - '[Term] is defined as [Definition]'
    - '[Term] refers to [Definition]'
    - '[Term] is a [Definition]'
    - '[Term] represents [Definition]'
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    sentences = split_into_sentences(cleaned)
    concepts: list[dict[str, str]] = []
    seen_terms = set()

    # Definition patterns: (term group, definition verb/phrase, definition group)
    definition_regexes = [
        re.compile(r"^\s*([A-Z][a-zA-Z0-9\s\-]{2,35})\s+(?:is|are)\s+defined\s+as\s+([^.;]+[.!]?)", re.IGNORECASE),
        re.compile(r"^\s*([A-Z][a-zA-Z0-9\s\-]{2,35})\s+refers\s+to\s+([^.;]+[.!]?)", re.IGNORECASE),
        re.compile(r"^\s*([A-Z][a-zA-Z0-9\s\-]{2,35})\s+(?:is|are)\s+(?:a|an|the)\s+([^.;]+(?:that|which|used|characterized|composed)[^.;]+[.!]?)", re.IGNORECASE),
        re.compile(r"^\s*([A-Z][a-zA-Z0-9\s\-]{2,35})\s+represents\s+([^.;]+[.!]?)", re.IGNORECASE),
        re.compile(r"^\s*([A-Z][a-zA-Z0-9\s\-]{2,35})\s+means\s+([^.;]+[.!]?)", re.IGNORECASE),
        re.compile(r"\b([A-Z][a-zA-Z0-9\s\-]{2,30}):\s+([A-Z][^.;]+[.!]?)"),  # Term: Definition format
    ]

    IGNORE_STRUCTURAL_TERMS = {
        "chapter", "unit", "section", "module", "part", "page",
        "figure", "fig", "table", "note", "example", "exercise",
        "homework", "syllabus", "author", "title", "version"
    }

    for sentence in sentences:
        s_clean = sentence.strip()
        for regex in definition_regexes:
            match = regex.search(s_clean)
            if match:
                term = match.group(1).strip()
                definition = match.group(2).strip()

                # Basic validation
                term_clean = term.title()
                term_words = term_clean.split()
                if (
                    1 <= len(term_words) <= 5
                    and len(definition) >= 15
                    and term_clean.lower() not in seen_terms
                    and not term_clean.lower().startswith(("in", "when", "although", "since", "as"))
                    and term_words[0].lower() not in IGNORE_STRUCTURAL_TERMS
                    and not any(w.lower() in IGNORE_STRUCTURAL_TERMS for w in term_words[:2])
                ):
                    seen_terms.add(term_clean.lower())
                    # Ensure definition ends with punctuation
                    if not definition.endswith((".", "!", "?")):
                        definition += "."
                    concepts.append({
                        "term": term_clean,
                        "definition": definition,
                    })
                    break
        if len(concepts) >= top_n:
            break

    # If few concepts matched strict definitions, extract key capitalized terms
    if len(concepts) < 3:
        for sentence in sentences:
            if len(concepts) >= top_n:
                break
            # Find technical phrases starting with a capital letter
            m = re.match(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(is|are|involves)\s+(.+)", sentence)
            if m:
                term = m.group(1).strip()
                term_words = term.title().split()
                if (
                    term.lower() not in seen_terms
                    and len(term_words) <= 3
                    and term_words[0].lower() not in IGNORE_STRUCTURAL_TERMS
                ):
                    seen_terms.add(term.lower())
                    definition = sentence.strip()
                    concepts.append({
                        "term": term,
                        "definition": definition,
                    })

    # If still fewer than 3 concepts, pair top keywords with their most descriptive sentence
    if len(concepts) < 3:
        keywords = extract_keywords(cleaned, top_n=top_n * 2)
        for kw in keywords:
            if len(concepts) >= top_n:
                break
            kw_lower = kw.lower()
            if kw_lower in seen_terms or any(term_w in IGNORE_STRUCTURAL_TERMS for term_w in kw_lower.split()):
                continue
            # Find best descriptive sentence containing this keyword
            for sentence in sentences:
                s_strip = sentence.strip()
                s_lower = s_strip.lower()
                if kw_lower in s_lower and 10 <= len(s_strip.split()) <= 45:
                    seen_terms.add(kw_lower)
                    concepts.append({
                        "term": kw.title(),
                        "definition": s_strip,
                    })
                    break

    return concepts


def extract_exam_points(text: str, max_points: int = 8) -> list[str]:
    """
    Extracts high-yield examination preparation points and core takeaways.
    Scores sentences based on indicators like 'important', 'crucial', 'key',
    'principle', 'formula', 'difference', 'advantage', 'remember', 'essential'.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    sentences = split_into_sentences(cleaned)
    if not sentences:
        return []

    exam_indicators = [
        "important", "crucial", "essential", "key", "fundamental", "vital",
        "principle", "rule", "law", "formula", "remember", "note that",
        "primary", "main reason", "difference between", "advantage", "disadvantage",
        "characteristic", "critical", "significant", "in conclusion", "must be",
    ]

    scored_sentences: list[tuple[float, str]] = []

    for sentence in sentences:
        s_clean = sentence.strip()
        word_count = len(s_clean.split())

        # Exclude very short or overly long sentences
        if word_count < 8 or word_count > 60:
            continue

        lower_s = s_clean.lower()
        score = 0.0

        # Check for exam indicators
        for ind in exam_indicators:
            if ind in lower_s:
                score += 2.5

        # Reward sentences with informative punctuation or numerical terms
        if re.search(r"\b\d+(?:\.\d+)?%?\b", s_clean):
            score += 0.8
        if ":" in s_clean or ";" in s_clean:
            score += 0.5

        # Penalize questions or conversational sentences
        if s_clean.endswith("?") or lower_s.startswith(("we will", "let us", "in this section")):
            score -= 2.0

        if score > 0:
            scored_sentences.append((score, s_clean))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    points: list[str] = []
    seen = set()

    for _, sentence in scored_sentences:
        # Avoid duplicate sentences
        first_few_words = " ".join(sentence.lower().split()[:6])
        if first_few_words not in seen:
            seen.add(first_few_words)
            points.append(sentence)
        if len(points) >= max_points:
            break

    # If not enough indicator sentences were found, take representative sentences
    if len(points) < 3 and len(sentences) >= 3:
        for s in sentences[:max_points]:
            if s not in points and 10 <= len(s.split()) <= 45:
                points.append(s.strip())
            if len(points) >= max_points:
                break

    return points


def extract_topics_and_subtopics(text: str, pages: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """
    Identifies main topics and subtopics from headings and structural cues in the document.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    topics: list[dict[str, Any]] = []
    current_topic: dict[str, Any] | None = None

    heading_patterns = [
        re.compile(r"^(?:Chapter|Unit|Module|Section)\s+\d+[:\s\-]*(.+)$", re.IGNORECASE),
        re.compile(r"^\d+\.\d*\s+([A-Z].+)$"),
        re.compile(r"^[I|V|X]+\.\s+([A-Z].+)$"),
        re.compile(r"^([A-Z\s]{4,40})$"),  # Short ALL-CAPS lines
    ]

    for line in lines:
        is_heading = False
        topic_title = ""

        for pattern in heading_patterns:
            m = pattern.match(line)
            if m:
                is_heading = True
                topic_title = m.group(1).strip() if m.groups() else line.strip()
                break

        if is_heading and len(topic_title) > 3 and not topic_title.isdigit():
            normalized_title = topic_title.title()
            # Avoid duplicate consecutive topics with identical titles
            if current_topic:
                if current_topic["topic"] != normalized_title:
                    topics.append(current_topic)
                    current_topic = {
                        "topic": normalized_title,
                        "subtopics": [],
                    }
            else:
                current_topic = {
                    "topic": normalized_title,
                    "subtopics": [],
                }
        elif current_topic and line.startswith(("-", "•", "*", "–")) and len(line) < 100:
            sub = line.lstrip("-•*– ").strip()
            if sub and len(sub) > 4 and sub not in current_topic["subtopics"]:
                current_topic["subtopics"].append(sub)

    if current_topic and (not topics or topics[-1]["topic"] != current_topic["topic"]):
        topics.append(current_topic)

    # Fallback: if no explicit headings found, generate topics from pages or high-yield clusters
    if not topics:
        if pages and len(pages) > 1:
            for page in pages[:6]:
                p_text = clean_text(page.get("text") or page.get("extracted_text", ""))
                p_sents = split_into_sentences(p_text)
                first_heading = p_sents[0][:60] if p_sents else f"Page {page.get('page_number', 1)}"
                topics.append({
                    "topic": f"Section (Page {page.get('page_number', 1)}): {first_heading}",
                    "subtopics": [s[:80] + "..." for s in p_sents[1:4]] if len(p_sents) > 1 else [],
                })
        else:
            keywords = extract_keywords(cleaned, top_n=4)
            for kw in keywords:
                topics.append({
                    "topic": f"Core Study Theme: {kw}",
                    "subtopics": [f"Key principles and applications of {kw.lower()}"],
                })

    return topics


# ------------------------------------------------------------------------------
# 4. Multi-Mode Summarization Coordinator (Task 3)
# ------------------------------------------------------------------------------

def generate_quick_summary(
    text: str,
    engine: SummarizerEngine | None = None,
) -> str:
    """
    Quick Summary Mode:
    Produces a crisp executive overview followed by 3-5 high-impact bulleted key points.
    Ideal for rapid review before a lecture or exam.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    if engine is None:
        engine = SummarizerEngine.get_instance()

    words = cleaned.split()
    if len(words) <= 40:
        return cleaned

    chunks = chunk_text(cleaned, max_chunk_words=380, overlap_words=30)
    chunk_summaries: list[str] = []

    for chunk in chunks[:3]:
        chunk_sum = engine.generate_summary_for_chunk(
            chunk,
            max_length=95,
            min_length=35,
            num_beams=2,
        )
        if chunk_sum:
            chunk_summaries.append(chunk_sum)

    combined_text = " ".join(chunk_summaries)
    if len(chunk_summaries) > 1:
        overview = engine.generate_summary_for_chunk(
            combined_text,
            max_length=110,
            min_length=45,
            num_beams=2,
        )
    else:
        overview = combined_text

    # Extract 3-5 high-yield exam points / key concepts for bulleted takeaways
    exam_pts = extract_exam_points(cleaned, max_points=4)
    if not exam_pts:
        concepts = extract_key_concepts(cleaned, top_n=3)
        if concepts:
            exam_pts = [f"{c['term']}: {c['definition']}" for c in concepts]
        else:
            all_sents = split_into_sentences(cleaned)
            exam_pts = all_sents[1:4] if len(all_sents) > 1 else []

    formatted_points: list[str] = []
    overview_sents = [s.lower() for s in split_into_sentences(overview)]
    for pt in exam_pts:
        pt_clean = pt.strip()
        first_few = " ".join(pt_clean.lower().split()[:5])
        if not any(first_few in osent for osent in overview_sents):
            if not pt_clean.endswith((".", "!", "?")):
                pt_clean += "."
            formatted_points.append(f"• {pt_clean}")
        if len(formatted_points) >= 4:
            break

    if formatted_points:
        return f"{overview}\n\nKey Points:\n" + "\n".join(formatted_points)
    return overview


def generate_standard_summary(
    text: str,
    engine: SummarizerEngine | None = None,
) -> str:
    """
    Standard Summary Mode:
    Generates a balanced, detailed explanation covering the main topics and narrative flow.
    Organized into structured paragraphs.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    if engine is None:
        engine = SummarizerEngine.get_instance()

    words = cleaned.split()
    if len(words) <= 45:
        return cleaned

    chunks = chunk_text(cleaned, max_chunk_words=380, overlap_words=35)
    chunk_summaries: list[str] = []

    for chunk in chunks:
        chunk_sum = engine.generate_summary_for_chunk(
            chunk,
            max_length=150,
            min_length=50,
            num_beams=2,
        )
        if chunk_sum:
            chunk_summaries.append(chunk_sum)

    if not chunk_summaries:
        return ""

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    # Combine chunk summaries into structured multi-paragraph narrative
    paragraphs: list[str] = []
    current_para: list[str] = []

    for i, c_sum in enumerate(chunk_summaries):
        current_para.append(c_sum)
        if len(current_para) >= 2 or i == len(chunk_summaries) - 1:
            paragraphs.append(" ".join(current_para))
            current_para = []

    return "\n\n".join(paragraphs)


def generate_detailed_summary(
    text: str,
    pages: list[dict[str, Any]] | None = None,
    engine: SummarizerEngine | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Detailed Summary Mode:
    Provides a comprehensive, in-depth explanation structured into:
    - Executive Overview
    - Topic-by-Topic Core Explanations
    - Page-by-page / Chapter-wise breakdown summaries
    Returns a tuple of: (structured_detailed_summary_text, chapter_summaries_list)
    """
    cleaned = clean_text(text)
    if not cleaned:
        return "", []

    if engine is None:
        engine = SummarizerEngine.get_instance()

    # 1. Executive overview
    standard_sum = generate_standard_summary(cleaned, engine=engine)

    # 2. Page-wise / Chapter-wise summaries
    chapter_summaries: list[dict[str, Any]] = []

    if pages and len(pages) > 0:
        for p in pages:
            p_text = clean_text(p.get("text") or p.get("extracted_text", ""))
            p_num = p.get("page_number", 1)
            p_words = len(p_text.split())

            if p_words < 3:
                continue

            if p_words < 20:
                page_sum = p_text
            else:
                page_sum = engine.generate_summary_for_chunk(
                    p_text,
                    max_length=120,
                    min_length=35,
                    num_beams=2,
                )
            chapter_summaries.append({
                "section": f"Page {p_num}",
                "page_number": p_num,
                "summary": page_sum,
                "word_count": len(page_sum.split()),
            })
    else:
        # If no page array provided, chunk into logical sections
        chunks = chunk_text(cleaned, max_chunk_words=380, overlap_words=20)
        for idx, chunk in enumerate(chunks, 1):
            c_sum = engine.generate_summary_for_chunk(
                chunk,
                max_length=130,
                min_length=40,
                num_beams=2,
            )
            chapter_summaries.append({
                "section": f"Section {idx}",
                "page_number": idx,
                "summary": c_sum,
                "word_count": len(c_sum.split()),
            })

    # Assemble structured detailed summary text
    sections: list[str] = [
        "### Executive Overview\n" + standard_sum,
        "### In-Depth Section Analysis",
    ]

    for item in chapter_summaries:
        sections.append(f"**{item['section']} Summary:**\n{item['summary']}")

    detailed_text = "\n\n".join(sections)
    return detailed_text, chapter_summaries


# ------------------------------------------------------------------------------
# 5. Master API Orchestrator (Task 1 & Task 5)
# ------------------------------------------------------------------------------

def generate_study_summary(
    content: str | list[dict[str, Any]],
    summary_type: str = "standard",
    document_metadata: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Main entry point for generating AI summaries and study materials.
    
    Args:
        content: Either the combined raw text or a list of page dicts [{"page_number": int, "text": str}]
        summary_type: 'quick', 'standard', or 'detailed'
        document_metadata: Optional dict containing document title, pages, etc.
        model_name: Optional model override (e.g. for testing)
        
    Returns:
        Dict containing:
        - success: bool
        - summary_type: str ('quick', 'standard', 'detailed')
        - summary_text: str
        - chapter_summaries: List[Dict[str, Any]]
        - key_concepts: List[Dict[str, str]]
        - keywords: List[str]
        - exam_points: List[str]
        - topics: List[Dict[str, Any]]
        - stats: Dict[str, Any]
        - model_name: str
        - error: Optional[str]
    """
    # 1. Unpack input and handle page lists vs raw text
    pages: list[dict[str, Any]] | None = None
    if isinstance(content, list):
        pages = content
        raw_text = "\n\n".join([p.get("text") or p.get("extracted_text", "") for p in pages if isinstance(p, dict)])
    elif isinstance(content, str):
        raw_text = content
    else:
        raise SummarizationError("Invalid content format. Expected string or list of page dictionaries.")

    # 2. Text integration & validation (Task 1)
    cleaned = clean_text(raw_text)
    total_words = len(cleaned.split())

    if not cleaned or total_words == 0:
        return {
            "success": False,
            "error": "The document contains no extractable text. Scanned or image-based PDFs cannot be summarized without OCR.",
            "summary_type": summary_type,
            "summary_text": "",
            "chapter_summaries": [],
            "key_concepts": [],
            "keywords": [],
            "exam_points": [],
            "topics": [],
            "stats": {"original_words": 0, "summary_words": 0, "reduction_percentage": 0.0},
            "model_name": model_name or "none",
        }

    # Normalize summary_type
    summary_type = (summary_type or "standard").lower().strip()
    if summary_type not in ("quick", "standard", "detailed"):
        summary_type = "standard"

    # Initialize AI Engine
    engine = SummarizerEngine.get_instance(model_name=model_name)

    # 3. Generate summary according to requested mode
    chapter_summaries: list[dict[str, Any]] = []

    if summary_type == "quick":
        summary_text = generate_quick_summary(cleaned, engine=engine)
    elif summary_type == "detailed":
        summary_text, chapter_summaries = generate_detailed_summary(cleaned, pages=pages, engine=engine)
    else:  # standard
        summary_text = generate_standard_summary(cleaned, engine=engine)

    # If detailed was not chosen, still generate lightweight chapter summaries if pages exist
    if not chapter_summaries and pages and len(pages) > 1:
        # Provide lightweight page breakdown
        for p in pages:
            p_text = clean_text(p.get("text") or p.get("extracted_text", ""))
            if len(p_text.split()) >= 3:
                sents = split_into_sentences(p_text)
                chapter_summaries.append({
                    "section": f"Page {p.get('page_number', 1)}",
                    "page_number": p.get("page_number", 1),
                    "summary": " ".join(sents[:2]),
                    "word_count": len(" ".join(sents[:2]).split()),
                })

    # 4. Extract concepts, keywords, exam points, and topics (Task 4)
    key_concepts = extract_key_concepts(cleaned, top_n=8)
    keywords = extract_keywords(cleaned, top_n=15)
    exam_points = extract_exam_points(cleaned, max_points=8)
    topics = extract_topics_and_subtopics(cleaned, pages=pages)

    # 5. Compute study metrics
    summary_words = len(summary_text.split())
    reduction = round((1.0 - (summary_words / max(total_words, 1))) * 100, 1) if total_words > summary_words else 0.0

    return {
        "success": True,
        "error": None,
        "summary_type": summary_type,
        "summary_text": summary_text,
        "chapter_summaries": chapter_summaries,
        "key_concepts": key_concepts,
        "keywords": keywords,
        "exam_points": exam_points,
        "topics": topics,
        "stats": {
            "original_words": total_words,
            "summary_words": summary_words,
            "reduction_percentage": max(0.0, reduction),
        },
        "model_name": engine.model_name,
    }
