"""
data_curation.py

Processes raw Amazon ESCI and Wayfair WANDS datasets into a unified format
suitable for OpenSearch ingestion (BM25 + HNSW) and offline evaluation.

SOLID design:
  S — Each class has a single, clearly bounded responsibility.
  O — New datasets can be added by subclassing DatasetProcessor; no existing
      code needs to change.
  L — ESCIProcessor and WANDSProcessor are interchangeable wherever a
      DatasetProcessor is expected.
  I — DatasetProcessor exposes only the methods callers actually need.
  D — DatasetMerger depends on the DatasetProcessor abstraction, not on
      ESCI or WANDS concretions.

Unified product document schema (ProductDocument):
  ┌─────────────────┬────────────┬───────────────────────────────────────────┐
  │ Field           │ OS type    │ Notes                                     │
  ├─────────────────┼────────────┼───────────────────────────────────────────┤
  │ product_id      │ keyword    │ prefixed: amz_<id> / wands_<id>           │
  │ source          │ keyword    │ "ESCI" | "WANDS"                          │
  │ full_text       │ text       │ enriched; title > brand > bullets > desc  │
  │ brand           │ keyword    │ normalized, lowercased                    │
  │ color           │ keyword[]  │ normalized list (split compound colors)   │
  │ product_class   │ keyword    │ high-level category                       │
  │ category_path   │ keyword    │ WANDS hierarchy; null for ESCI            │
  │ average_rating  │ float      │ null for ESCI (not in dataset)            │
  │ review_count    │ integer    │ null for ESCI (not in dataset)            │
  │ metadata        │ object     │ full raw fields preserved for debugging   │
  └─────────────────┴────────────┴───────────────────────────────────────────┘

Usage (in database_ingestion.ipynb):
    from product_search.data_curation import ESCIProcessor, WANDSProcessor, DatasetMerger

    esci  = ESCIProcessor(amz_train_df, amz_test_df)
    wands = WANDSProcessor(wands_train_df, wands_test_df)

    merger   = DatasetMerger([esci, wands])
    artifacts = merger.build()

    # artifacts.product_store       → {product_id: ProductDocument-as-dict}
    # artifacts.train_qrels_dict    → {query_id: {product_id: gain}}
    # artifacts.test_qrels_dict     → same
    # artifacts.train_query_table   → pd.DataFrame [query_id, query]
    # artifacts.test_query_table    → pd.DataFrame [query_id, query]
"""

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


# ============================================================================
# 1. Value objects / data containers
# ============================================================================

@dataclass
class ProductDocument:
    """
    Unified product representation shared across all datasets.

    Stored as the value in product_store dicts and ultimately ingested into
    both the BM25 and HNSW OpenSearch indices.

    Design note: keeping this as a dataclass (rather than a plain dict) gives
    us type safety during construction and a clean .to_dict() for serialisation.
    """
    product_id: str
    source: str                         # "ESCI" | "WANDS"
    full_text: str                      # enriched text for BM25 (title+brand+bullets+description)
    encode_text: str                    # short text for dense embedding (title+brand only)
    brand: Optional[str]                # keyword filter
    color: List[str]                    # list of normalised colour tokens
    product_class: Optional[str]        # high-level category keyword
    category_path: Optional[str]        # WANDS slash-delimited hierarchy
    average_rating: Optional[float]     # numeric field (WANDS only)
    review_count: Optional[int]         # numeric field (WANDS only)
    metadata: Dict[str, Any]            # raw fields, preserved for debugging

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict suitable for JSON / OpenSearch ingestion."""
        return {
            "product_id":     self.product_id,
            "source":         self.source,
            "full_text":      self.full_text,
            "encode_text":    self.encode_text,
            "brand":          self.brand,
            "color":          self.color,
            "product_class":  self.product_class,
            "category_path":  self.category_path,
            "average_rating": self.average_rating,
            "review_count":   self.review_count,
            "metadata":       self.metadata,
        }


@dataclass
class DatasetArtifacts:
    """
    All artefacts produced by a single DatasetProcessor.

    Kept as a dataclass so DatasetMerger can merge multiple instances without
    knowing anything about the source dataset.
    """
    # {product_id: ProductDocument}
    product_store: Dict[str, ProductDocument] = field(default_factory=dict)

    # {query_id: {product_id: gain}}
    train_qrels_dict: Dict[str, Dict[str, float]] = field(default_factory=dict)
    test_qrels_dict: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # DataFrame columns: [query_id, query]
    train_query_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_query_table: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class MergedArtifacts:
    """
    The final merged output from DatasetMerger, ready for ingestion.

    product_store values are plain dicts (via ProductDocument.to_dict()) so
    they can be JSON-serialised and fed directly to bulk_ingest_bm25 /
    bulk_ingest_hnsw without further transformation.
    """
    # {product_id: dict}  ← serialised ProductDocument
    product_store: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    train_qrels_dict: Dict[str, Dict[str, float]] = field(default_factory=dict)
    test_qrels_dict: Dict[str, Dict[str, float]] = field(default_factory=dict)

    train_query_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_query_table: pd.DataFrame = field(default_factory=pd.DataFrame)


# ============================================================================
# 2. AttributeNormalizer  (SRP: only knows how to clean attribute strings)
# ============================================================================

class AttributeNormalizer:
    """
    Stateless helper methods for cleaning and normalising product attributes.

    All methods are class methods so callers don't need to instantiate this
    class — it acts as a namespace for related transformations.
    """

    # Base colour vocabulary used to filter compound marketing colour strings.
    # Extend this set freely; it does not affect any other class.
    BASE_COLORS: Set[str] = {
        "red", "blue", "green", "black", "white", "grey", "gray", "brown",
        "yellow", "pink", "purple", "orange", "silver", "gold", "beige",
        "navy", "teal", "ivory", "cream", "tan", "maroon", "coral", "cyan",
        "magenta", "turquoise", "charcoal", "multicolor", "multi",
    }

    # Tokens that appear in colour strings but convey no colour information.
    _COLOR_NOISE: Set[str] = {
        "color", "colour", "finish", "shade", "tone", "hue",
        "light", "dark", "deep", "bright", "matte", "glossy",
        "and", "with", "mix", "blend",
    }

    @classmethod
    def normalize_brand(cls, raw: Any) -> Optional[str]:
        """
        Lowercase and strip a brand name.
        Returns None for null / 'nan' / empty strings.
        """
        if raw is None:
            return None
        s = str(raw).strip().lower()
        if not s or s == "nan":
            return None
        return s

    @classmethod
    def normalize_color(cls, raw: Any) -> List[str]:
        """
        Parse a raw colour string into a list of normalised base colour tokens.

        Handles:
          - Null / 'nan' → []
          - Single colours:     "Black"          → ["black"]
          - Compound colours:   "Red/Blue"        → ["red", "blue"]
          - Marketing names:    "Space Gray"      → ["gray"]
          - Multi-value lists:  "Red, Navy, Gold" → ["red", "navy", "gold"]

        Only tokens present in BASE_COLORS are kept, so marketing adjectives
        like "Deep" or "Bright" are stripped while the colour token is kept.
        """
        if raw is None:
            return []
        s = str(raw).strip().lower()
        if not s or s == "nan":
            return []

        # Split on common delimiters used in compound colour names
        tokens = re.split(r"[/,|&\-\s]+", s)
        colors: List[str] = []
        for tok in tokens:
            tok = tok.strip()
            if tok and tok not in cls._COLOR_NOISE and tok in cls.BASE_COLORS:
                if tok not in colors:
                    colors.append(tok)
        return colors

    @classmethod
    def normalize_rating(cls, raw: Any) -> Optional[float]:
        """Parse a rating value to float; return None on failure."""
        if raw is None:
            return None
        try:
            v = float(raw)
            return v if 0.0 <= v <= 5.0 else None
        except (ValueError, TypeError):
            return None

    @classmethod
    def normalize_count(cls, raw: Any) -> Optional[int]:
        """Parse a count value (review_count etc.) to int; return None on failure."""
        if raw is None:
            return None
        try:
            return int(float(raw))
        except (ValueError, TypeError):
            return None

    @classmethod
    def parse_wands_features(cls, raw: Any) -> Dict[str, str]:
        """
        Parse WANDS pipe-delimited 'attribute:value' feature string into a dict.

        Example input:  "Color:Gray | Material:Wood | Width:36 inches"
        Example output: {"color": "gray", "material": "wood", "width": "36 inches"}

        Keys are lowercased and stripped. Values are stripped but not lowercased
        (dimensions and model numbers are case-sensitive).
        """
        if raw is None:
            return {}
        s = str(raw).strip()
        if not s or s.lower() == "nan":
            return {}

        result: Dict[str, str] = {}
        for pair in s.split("|"):
            pair = pair.strip()
            if ":" not in pair:
                continue
            key, _, value = pair.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if key and value:
                result[key] = value
        return result

    @classmethod
    def normalize_str(cls, raw: Any) -> Optional[str]:
        """Strip and return a string; None for null / 'nan' / empty."""
        if raw is None:
            return None
        s = str(raw).strip()
        return s if s and s.lower() != "nan" else None


# ============================================================================
# 3. FullTextBuilder  (SRP: only builds full_text strings)
# ============================================================================

class FullTextBuilder:
    """
    Constructs an enriched `full_text` string from product metadata.

    Field ordering follows descending BM25 signal strength:
      1. Product title / name      — highest signal
      2. Brand                     — exact-match queries
      3. Product class / category  — category-level queries
      4. Bullet points / features  — attribute-level queries (machine washable, etc.)
      5. Description               — semantic fallback

    Design note: this class knows nothing about ESCI or WANDS schema — it
    receives a plain dict and a prioritised field list, making it easy to
    extend for new datasets without modification (OCP).
    """

    @staticmethod
    def build(fields_in_order: List[Tuple[str, Any]]) -> str:
        """
        Build a single full_text string from an ordered list of (label, value)
        pairs. Null / empty / 'nan' values are skipped automatically.

        Args:
            fields_in_order: ordered list of (field_name, raw_value).
                The field_name is ignored in the output — only the value matters.
                Ordering determines BM25 term frequency weighting.

        Returns:
            A single whitespace-joined string with deduped segments.
        """
        seen: Set[str] = set()
        parts: List[str] = []
        for _, value in fields_in_order:
            if value is None:
                continue
            s = str(value).strip()
            if not s or s.lower() == "nan":
                continue
            # Avoid repeating the exact same segment (e.g. title == name)
            if s not in seen:
                seen.add(s)
                parts.append(s)
        return " ".join(parts).strip()

    @staticmethod
    def build_from_esci(meta: Dict[str, Any]) -> str:
        """
        ESCI full_text for BM25: title > brand > bullet_points > description.
        """
        bullets_raw = meta.get("product_bullet_point")
        bullets = re.sub(r"[\|\n\r]+", " ", str(bullets_raw)).strip() if bullets_raw else ""

        return FullTextBuilder.build([
            ("title",       meta.get("product_title")),
            ("brand",       meta.get("product_brand")),
            ("bullets",     bullets or None),
            ("description", meta.get("product_description")),
        ])

    @staticmethod
    def build_from_wands(meta: Dict[str, Any], parsed_features: Dict[str, str]) -> str:
        """
        WANDS full_text for BM25: name > class > category > features > description.
        """
        feature_text = " ".join(parsed_features.values()) if parsed_features else ""

        return FullTextBuilder.build([
            ("name",         meta.get("product_name")),
            ("class",        meta.get("product_class")),
            ("category",     meta.get("category hierarchy")),
            ("features",     feature_text or None),
            ("description",  meta.get("product_description")),
        ])

    @staticmethod
    def encode_text_from_esci(meta: Dict[str, Any], colors: Optional[List[str]] = None) -> str:
        """
        Short text for dense embedding (ESCI): title + brand + color tokens.
        Color is appended so color-specific queries (e.g. "teal tutu") can
        match even when the color word isn't in the product title.
        """
        color_str = " ".join(colors) if colors else None
        return FullTextBuilder.build([
            ("title", meta.get("product_title")),
            ("brand", meta.get("product_brand")),
            ("color", color_str),
        ])

    @staticmethod
    def encode_text_from_wands(meta: Dict[str, Any], colors: Optional[List[str]] = None) -> str:
        """
        Short text for dense embedding (WANDS): name + class + color tokens.
        """
        color_str = " ".join(colors) if colors else None
        return FullTextBuilder.build([
            ("name",  meta.get("product_name")),
            ("class", meta.get("product_class")),
            ("color", color_str),
        ])


# ============================================================================
# 4. DatasetProcessor  (abstract interface — DIP + LSP)
# ============================================================================

class DatasetProcessor(ABC):
    """
    Abstract base class defining the contract for all dataset processors.

    Any class that processes a raw dataset into the unified schema must:
      1. Accept train and test raw DataFrames at construction time.
      2. Implement process() to return a DatasetArtifacts instance.

    This abstraction is what DatasetMerger depends on, ensuring that ESCI,
    WANDS, or any future dataset can be swapped in without touching the merger.
    """

    # Subclasses declare these constants to make grading transparent.
    GRADING: Dict[str, float] = {}
    SOURCE_NAME: str = ""
    ID_PREFIX: str = ""

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        keep_irrelevant: bool = False,
        top_k_per_query: Optional[int] = 40,
    ) -> None:
        """
        Args:
            train_df:          Raw merged training DataFrame.
            test_df:           Raw merged test DataFrame.
            keep_irrelevant:   If True, include gain=0 docs in qrels.
                               Usually False — irrelevant docs inflate qrel size.
            top_k_per_query:   Cap the number of judged docs per query.
                               Keeps qrel files manageable for large datasets.
        """
        self._train_df = train_df.copy()
        self._test_df = test_df.copy()
        self._keep_irrelevant = keep_irrelevant
        self._top_k_per_query = top_k_per_query

    @abstractmethod
    def process(self) -> DatasetArtifacts:
        """
        Process raw DataFrames into a unified DatasetArtifacts instance.

        Subclasses must implement all four artefacts:
          - product_store
          - train_qrels_dict / test_qrels_dict
          - train_query_table / test_query_table
        """
        ...

    # ------------------------------------------------------------------
    # Shared utilities available to all subclasses
    # ------------------------------------------------------------------

    def _prefix_id(self, raw_id: Any) -> str:
        return f"{self.ID_PREFIX}{raw_id}"

    def _build_qrels_dict(
        self,
        df: pd.DataFrame,
        *,
        query_id_col: str,
        product_id_col: str,
        label_col: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Generic qrels builder shared by all processors.

        Steps:
          1. Map label strings → float gains using self.GRADING
          2. Drop unknowns / nulls
          3. Optionally drop gain=0 (irrelevant)
          4. Dedupe (qid, pid) keeping max gain
          5. Cap at top_k_per_query if set
          6. Return {qid: {pid: gain}}
        """
        work = df[[query_id_col, product_id_col, label_col]].copy()
        work[query_id_col]   = work[query_id_col].apply(lambda x: self._prefix_id(x))
        work[product_id_col] = work[product_id_col].apply(lambda x: self._prefix_id(x))

        work["_label_upper"] = work[label_col].astype(str).str.strip().str.upper()
        work["_gain"] = work["_label_upper"].map(self.GRADING)
        work = work.dropna(subset=["_gain", query_id_col, product_id_col])

        if not self._keep_irrelevant:
            work = work[work["_gain"] > 0.0]

        # Deduplicate keeping max gain
        work = (
            work.sort_values("_gain", ascending=False, kind="mergesort")
                .drop_duplicates(subset=[query_id_col, product_id_col], keep="first")
        )

        # Sort for determinism
        work = work.sort_values(
            [query_id_col, "_gain", product_id_col],
            ascending=[True, False, True],
            kind="mergesort",
        )

        if self._top_k_per_query is not None:
            work = work.groupby(query_id_col, sort=False, as_index=False).head(
                self._top_k_per_query
            )

        qrels: Dict[str, Dict[str, float]] = {}
        for qid, sub in work.groupby(query_id_col, sort=False):
            qrels[str(qid)] = {
                str(pid): float(g)
                for pid, g in zip(sub[product_id_col], sub["_gain"])
            }
        return qrels

    def _build_query_table(
        self,
        df: pd.DataFrame,
        *,
        query_id_col: str,
        query_col: str,
    ) -> pd.DataFrame:
        """Build a deduplicated [query_id, query] DataFrame."""
        qt = (
            df[[query_id_col, query_col]]
            .drop_duplicates(subset=[query_id_col], keep="first")
            .copy()
        )
        qt[query_id_col] = qt[query_id_col].apply(lambda x: self._prefix_id(x))
        return (
            qt.rename(columns={query_id_col: "query_id", query_col: "query"})
              .reset_index(drop=True)
        )


# ============================================================================
# 5. ESCIProcessor  (SRP: only knows Amazon ESCI schema)
# ============================================================================

class ESCIProcessor(DatasetProcessor):
    """
    Processes the Amazon Shopping Queries (ESCI) dataset.

    Expected input DataFrame columns (after examples + products merge):
        query_id, query, product_id, esci_label,
        product_title, product_description, product_bullet_point,
        product_brand, product_color

    Grading:
        Exact (E)       → 3.0   (satisfies all query requirements)
        Substitute (S)  → 2.0   (partially satisfies, functional substitute)
        Complement (C)  → 1.0   (used alongside an exact match)
        Irrelevant (I)  → 0.0   (excluded by default)

    Note on price: ESCI does not include a price field. The max_price filter
    extracted by the LLM agent will be a no-op against this dataset.
    """

    SOURCE_NAME = "ESCI"
    ID_PREFIX   = "amz_"
    GRADING: Dict[str, float] = {
        "EXACT": 3.0, "E": 3.0,
        "SUBSTITUTE": 2.0, "S": 2.0,
        "COMPLEMENT": 1.0, "C": 1.0,
        "IRRELEVANT": 0.0, "I": 0.0,
    }

    # Columns preserved verbatim in ProductDocument.metadata
    _METADATA_COLS = [
        "product_title",
        "product_description",
        "product_bullet_point",
        "product_brand",
        "product_color",
    ]

    def process(self) -> DatasetArtifacts:
        """Build all four artefacts from train and test DataFrames."""
        train_qrels = self._build_qrels_dict(
            self._train_df,
            query_id_col="query_id",
            product_id_col="product_id",
            label_col="esci_label",
        )
        test_qrels = self._build_qrels_dict(
            self._test_df,
            query_id_col="query_id",
            product_id_col="product_id",
            label_col="esci_label",
        )
        train_qt = self._build_query_table(
            self._train_df,
            query_id_col="query_id",
            query_col="query",
        )
        test_qt = self._build_query_table(
            self._test_df,
            query_id_col="query_id",
            query_col="query",
        )

        # Build product store from union of train + test products
        combined = pd.concat([self._train_df, self._test_df], ignore_index=True)
        product_store = self._build_product_store(combined)

        return DatasetArtifacts(
            product_store=product_store,
            train_qrels_dict=train_qrels,
            test_qrels_dict=test_qrels,
            train_query_table=train_qt,
            test_query_table=test_qt,
        )

    def _build_product_store(self, df: pd.DataFrame) -> Dict[str, ProductDocument]:
        """
        Deduplicate by product_id and build one ProductDocument per product.
        """
        meta_cols = [c for c in self._METADATA_COLS if c in df.columns]
        prod_df = (
            df[["product_id"] + meta_cols]
            .drop_duplicates(subset=["product_id"], keep="first")
        )

        store: Dict[str, ProductDocument] = {}
        for row in prod_df.itertuples(index=False):
            meta = {c: AttributeNormalizer.normalize_str(getattr(row, c, None))
                    for c in meta_cols}

            pid = self._prefix_id(row.product_id)
            colors = AttributeNormalizer.normalize_color(meta.get("product_color"))
            doc = ProductDocument(
                product_id=pid,
                source=self.SOURCE_NAME,
                full_text=FullTextBuilder.build_from_esci(meta),
                encode_text=FullTextBuilder.encode_text_from_esci(meta, colors),
                brand=AttributeNormalizer.normalize_brand(meta.get("product_brand")),
                color=colors,
                product_class=None,           # ESCI has no product class field
                category_path=None,           # ESCI has no category hierarchy
                average_rating=None,          # ESCI has no rating in small version
                review_count=None,            # ESCI has no review count
                metadata=meta,
            )
            store[pid] = doc

        return store


# ============================================================================
# 6. WANDSProcessor  (SRP: only knows Wayfair WANDS schema)
# ============================================================================

class WANDSProcessor(DatasetProcessor):
    """
    Processes the Wayfair WANDS dataset.

    Expected input DataFrame columns (after query + label + product merge):
        query_id, query, query_class, product_id, label,
        product_name, product_class, category hierarchy,
        product_description, product_features,
        rating_count, average_rating, review_count

    Grading:
        Exact       → 2.0   (fully satisfies the query)
        Partial     → 1.0   (partially relevant)
        Irrelevant  → 0.0   (excluded by default)

    Key difference from ESCI: product attributes (color, material, style,
    dimensions) are encoded in the pipe-delimited `product_features` field
    as "Attribute:Value | Attribute:Value" pairs. These are parsed by
    AttributeNormalizer.parse_wands_features() into a structured dict,
    which is then used for both full_text enrichment and colour extraction.
    """

    SOURCE_NAME = "WANDS"
    ID_PREFIX   = "wands_"
    GRADING: Dict[str, float] = {
        "EXACT": 2.0,
        "PARTIAL": 1.0,
        "IRRELEVANT": 0.0,
    }

    # Columns preserved verbatim in ProductDocument.metadata
    _METADATA_COLS = [
        "product_name",
        "product_class",
        "category hierarchy",
        "product_description",
        "product_features",
        "rating_count",
        "average_rating",
        "review_count",
    ]

    def process(self) -> DatasetArtifacts:
        """Build all four artefacts from train and test DataFrames."""
        train_qrels = self._build_qrels_dict(
            self._train_df,
            query_id_col="query_id",
            product_id_col="product_id",
            label_col="label",
        )
        test_qrels = self._build_qrels_dict(
            self._test_df,
            query_id_col="query_id",
            product_id_col="product_id",
            label_col="label",
        )
        train_qt = self._build_query_table(
            self._train_df,
            query_id_col="query_id",
            query_col="query",
        )
        test_qt = self._build_query_table(
            self._test_df,
            query_id_col="query_id",
            query_col="query",
        )

        combined = pd.concat([self._train_df, self._test_df], ignore_index=True)
        product_store = self._build_product_store(combined)

        return DatasetArtifacts(
            product_store=product_store,
            train_qrels_dict=train_qrels,
            test_qrels_dict=test_qrels,
            train_query_table=train_qt,
            test_query_table=test_qt,
        )

    def _build_product_store(self, df: pd.DataFrame) -> Dict[str, ProductDocument]:
        """
        Deduplicate by product_id and build one ProductDocument per product.

        Parses product_features into a structured dict, then uses it for:
          - full_text enrichment
          - color extraction (looks for "Color" key in parsed features)
        """
        meta_cols = [c for c in self._METADATA_COLS if c in df.columns]
        prod_df = (
            df[["product_id"] + meta_cols]
            .drop_duplicates(subset=["product_id"], keep="first")
        )

        store: Dict[str, ProductDocument] = {}
        for row in prod_df.itertuples(index=False):
            raw_meta = {c: getattr(row, c, None) for c in meta_cols}

            # Parse structured features first — used by both full_text and color
            parsed_features = AttributeNormalizer.parse_wands_features(
                raw_meta.get("product_features")
            )

            # Extract color from parsed features (prefer "color" key)
            raw_color = parsed_features.get("color") or parsed_features.get("colour")
            colors = AttributeNormalizer.normalize_color(raw_color)

            # Clean metadata for storage (preserve raw, but handle NaN)
            meta = {
                c: (None if _is_null(raw_meta[c]) else raw_meta[c])
                for c in meta_cols
            }
            # Add parsed features dict to metadata for transparency
            meta["parsed_features"] = parsed_features

            # Normalise category path: "Furniture / Sofas / Sectionals"
            category_path = AttributeNormalizer.normalize_str(
                raw_meta.get("category hierarchy")
            )

            pid = self._prefix_id(row.product_id)
            doc = ProductDocument(
                product_id=pid,
                source=self.SOURCE_NAME,
                full_text=FullTextBuilder.build_from_wands(raw_meta, parsed_features),
                encode_text=FullTextBuilder.encode_text_from_wands(raw_meta, colors),
                brand=None,                   # WANDS has no brand field
                color=colors,
                product_class=AttributeNormalizer.normalize_str(
                    raw_meta.get("product_class")
                ),
                category_path=category_path,
                average_rating=AttributeNormalizer.normalize_rating(
                    raw_meta.get("average_rating")
                ),
                review_count=AttributeNormalizer.normalize_count(
                    raw_meta.get("review_count")
                ),
                metadata=meta,
            )
            store[pid] = doc

        return store


# ============================================================================
# 7. DatasetMerger  (OCP + DIP: depends only on DatasetProcessor interface)
# ============================================================================

class DatasetMerger:
    """
    Merges artefacts from any number of DatasetProcessor instances into a
    single unified MergedArtifacts object ready for OpenSearch ingestion.

    Design notes:
      - Depends on DatasetProcessor (the abstraction), not ESCI or WANDS.
      - Adding a third dataset requires only a new DatasetProcessor subclass;
        this class never needs to change (Open/Closed).
      - Query table deduplication uses (query_id, query) — prefixed IDs ensure
        no collisions across datasets.
      - product_store collision on the same product_id is logged as a warning
        and the first-seen entry wins (deterministic).
    """

    def __init__(self, processors: List[DatasetProcessor]) -> None:
        """
        Args:
            processors: list of initialised DatasetProcessor instances.
                        Order only matters for collision tie-breaking.
        """
        if not processors:
            raise ValueError("DatasetMerger requires at least one DatasetProcessor.")
        self._processors = processors

    def build(self) -> MergedArtifacts:
        """
        Run all processors and merge their artefacts.

        Returns a MergedArtifacts instance with:
          - product_store:        serialised dicts (ProductDocument.to_dict())
          - train/test_qrels:     merged, no key collisions (prefixed IDs)
          - train/test_query_table: concatenated + deduplicated DataFrames
        """
        all_artifacts = [p.process() for p in self._processors]

        merged_product_store: Dict[str, Dict[str, Any]] = {}
        merged_train_qrels:   Dict[str, Dict[str, float]] = {}
        merged_test_qrels:    Dict[str, Dict[str, float]] = {}
        train_query_tables:   List[pd.DataFrame] = []
        test_query_tables:    List[pd.DataFrame] = []

        for artifacts in all_artifacts:
            # Product store — serialise ProductDocument → dict
            for pid, doc in artifacts.product_store.items():
                if pid in merged_product_store:
                    print(f"[DatasetMerger][WARN] Duplicate product_id '{pid}' — keeping first.")
                    continue
                merged_product_store[pid] = doc.to_dict()

            # Qrels — prefixed IDs guarantee no cross-dataset collisions
            _merge_dicts_warn(merged_train_qrels, artifacts.train_qrels_dict, label="train_qrels")
            _merge_dicts_warn(merged_test_qrels,  artifacts.test_qrels_dict,  label="test_qrels")

            # Query tables
            if not artifacts.train_query_table.empty:
                train_query_tables.append(artifacts.train_query_table)
            if not artifacts.test_query_table.empty:
                test_query_tables.append(artifacts.test_query_table)

        merged_train_qt = _concat_query_tables(train_query_tables)
        merged_test_qt  = _concat_query_tables(test_query_tables)

        self._print_summary(merged_product_store, merged_train_qrels, merged_test_qrels,
                            merged_train_qt, merged_test_qt)

        return MergedArtifacts(
            product_store=merged_product_store,
            train_qrels_dict=merged_train_qrels,
            test_qrels_dict=merged_test_qrels,
            train_query_table=merged_train_qt,
            test_query_table=merged_test_qt,
        )

    def build_dev_sample(
        self,
        *,
        n_train_per_source: int = 1000,
        n_test_per_source: int = 250,
        n_distractor_per_source: int = 250,
        random_state: int = 42,
    ) -> "MergedArtifacts":
        """
        Build a small development sample suitable for running on a MacBook Air.

        Samples equally from each source (processor):
          - n_train_per_source train queries  →  default: 1 000 ESCI + 1 000 WANDS = 2 000
          - n_test_per_source  test queries   →  default:   250 ESCI +   250 WANDS =   500

        The product store contains only:
          1. Products judged against the sampled train + test queries (keeps evaluation honest).
          2. n_distractor_per_source additional products per source that are NOT in any sampled
             qrel — these act as realistic retrieval noise without bloating the store.

        Args:
            n_train_per_source:    Train queries to sample per processor. Default 1 000.
            n_test_per_source:     Test queries to sample per processor. Default 250.
            n_distractor_per_source: Extra non-judged products per processor. Default 250.
            random_state:          Seed for reproducibility. Default 42.

        Returns:
            MergedArtifacts with serialised product_store (dicts, not ProductDocuments),
            ready for JSON export and OpenSearch ingestion.
        """
        rng = random.Random(random_state)

        merged_product_store: Dict[str, Dict[str, Any]] = {}
        merged_train_qrels:   Dict[str, Dict[str, float]] = {}
        merged_test_qrels:    Dict[str, Dict[str, float]] = {}
        train_query_tables:   List[pd.DataFrame] = []
        test_query_tables:    List[pd.DataFrame] = []

        for proc in self._processors:
            artifacts = proc.process()

            # ── 1. Sample queries ─────────────────────────────────────────────
            all_train_qids = list(artifacts.train_qrels_dict.keys())
            all_test_qids  = list(artifacts.test_qrels_dict.keys())

            sampled_train_qids: Set[str] = set(
                rng.sample(all_train_qids, min(n_train_per_source, len(all_train_qids)))
            )
            sampled_test_qids: Set[str] = set(
                rng.sample(all_test_qids, min(n_test_per_source, len(all_test_qids)))
            )

            sampled_train_qrels = {
                qid: v for qid, v in artifacts.train_qrels_dict.items()
                if qid in sampled_train_qids
            }
            sampled_test_qrels = {
                qid: v for qid, v in artifacts.test_qrels_dict.items()
                if qid in sampled_test_qids
            }

            # ── 2. Collect all products judged in the sampled qrels ───────────
            judged_pids: Set[str] = set()
            for rel in sampled_train_qrels.values():
                judged_pids.update(rel.keys())
            for rel in sampled_test_qrels.values():
                judged_pids.update(rel.keys())

            # ── 3. Sample distractor products (not in any sampled qrel) ───────
            distractor_pool = [
                pid for pid in artifacts.product_store
                if pid not in judged_pids
            ]
            n_dist = min(n_distractor_per_source, len(distractor_pool))
            distractor_pids: Set[str] = set(rng.sample(distractor_pool, n_dist))

            # ── 4. Build product store slice (serialise ProductDocument → dict) ─
            for pid in judged_pids | distractor_pids:
                doc = artifacts.product_store.get(pid)
                if doc is not None and pid not in merged_product_store:
                    merged_product_store[pid] = doc.to_dict()

            # ── 5. Merge qrels (prefixed IDs guarantee no cross-source collisions)
            _merge_dicts_warn(merged_train_qrels, sampled_train_qrels, label="train_qrels")
            _merge_dicts_warn(merged_test_qrels,  sampled_test_qrels,  label="test_qrels")

            # ── 6. Filter query tables to sampled query IDs ───────────────────
            if not artifacts.train_query_table.empty:
                qt = artifacts.train_query_table[
                    artifacts.train_query_table["query_id"].isin(sampled_train_qids)
                ]
                if not qt.empty:
                    train_query_tables.append(qt.copy())

            if not artifacts.test_query_table.empty:
                qt = artifacts.test_query_table[
                    artifacts.test_query_table["query_id"].isin(sampled_test_qids)
                ]
                if not qt.empty:
                    test_query_tables.append(qt.copy())

        merged_train_qt = _concat_query_tables(train_query_tables)
        merged_test_qt  = _concat_query_tables(test_query_tables)

        self._print_summary(
            merged_product_store, merged_train_qrels, merged_test_qrels,
            merged_train_qt, merged_test_qt,
        )

        return MergedArtifacts(
            product_store=merged_product_store,
            train_qrels_dict=merged_train_qrels,
            test_qrels_dict=merged_test_qrels,
            train_query_table=merged_train_qt,
            test_query_table=merged_test_qt,
        )

    @staticmethod
    def _print_summary(
        product_store: Dict,
        train_qrels: Dict,
        test_qrels: Dict,
        train_qt: pd.DataFrame,
        test_qt: pd.DataFrame,
    ) -> None:
        sources = {}
        for doc in product_store.values():
            src = doc.get("source", "UNKNOWN")
            sources[src] = sources.get(src, 0) + 1

        print("\n[DatasetMerger] ── Merge summary ──────────────────────────")
        print(f"  Total products  : {len(product_store):>8,}")
        for src, cnt in sorted(sources.items()):
            print(f"    {src:<10}  : {cnt:>8,}")
        print(f"  Train queries   : {len(train_qrels):>8,}")
        print(f"  Test  queries   : {len(test_qrels):>8,}")
        print(f"  Train qrel docs : {sum(len(v) for v in train_qrels.values()):>8,}")
        print(f"  Test  qrel docs : {sum(len(v) for v in test_qrels.values()):>8,}")
        print("─────────────────────────────────────────────────────────────\n")


# ============================================================================
# 8. Private helpers
# ============================================================================

def _is_null(v: Any) -> bool:
    """True for None, NaN, and string 'nan'."""
    if v is None:
        return True
    try:
        import math
        return math.isnan(float(v)) if not isinstance(v, str) else str(v).lower() == "nan"
    except (ValueError, TypeError):
        return str(v).lower() == "nan"


def _merge_dicts_warn(
    target: Dict[str, Any],
    source: Dict[str, Any],
    label: str,
) -> None:
    """Merge source into target, printing a warning on key collisions."""
    for k, v in source.items():
        if k in target:
            print(f"[DatasetMerger][WARN] Duplicate key '{k}' in {label} — keeping first.")
        else:
            target[k] = v


def _concat_query_tables(tables: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate query table DataFrames and deduplicate by query_id."""
    if not tables:
        return pd.DataFrame(columns=["query_id", "query"])
    return (
        pd.concat(tables, ignore_index=True)
          .drop_duplicates(subset=["query_id"], keep="first")
          .reset_index(drop=True)
    )