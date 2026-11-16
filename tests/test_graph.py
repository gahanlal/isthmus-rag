"""Test bounded graph traversal, relationship handling, and evidence deduplication."""

from dataclasses import replace

from isthmus.graph import AdaptiveGraphRetriever, TraversalConfig
from isthmus.models import (
    Chunk,
    QueryPlan,
    QueryRequirement,
    Relationship,
    RelationshipType,
    SearchHit,
)


def setup_graph(scores, edges, document_id="doc"):
    """Build a graph fixture and return its chunks, edges, and retriever."""
    chunks = [
        Chunk(
            id=chunk_id,
            document_id=document_id,
            sector_id="sector",
            content=f"chunk {chunk_id}",
            sequence_number=number,
            token_count=1,
        )
        for number, (chunk_id, _score) in enumerate(scores, start=1)
    ]
    score_map = dict(scores)
    retriever = AdaptiveGraphRetriever(
        chunks,
        edges,
        similarity=lambda _query, chunk: score_map[chunk.id],
        config=TraversalConfig(
            max_nodes=20,
            max_depth=10,
            max_low_similarity_bridges=1,
            max_backward=20,
            max_forward=20,
            include_threshold=0.6,
        ),
    )
    plan = QueryPlan(
        "policy exception",
        "factual",
        (
            QueryRequirement("main", "policy", frozenset({"policy"})),
            QueryRequirement("exceptions", "exceptions", frozenset({"exception"})),
        ),
        True,
        True,
    )
    hits = [SearchHit(next(chunk for chunk in chunks if chunk.id == chunk_id), score, rank)
            for rank, (chunk_id, score) in enumerate(scores, start=1)
            if chunk_id == "seed"]
    return retriever, plan, hits


def sequential_edges(ids):
    """Create forward and backward relationships between consecutive chunks."""
    edges = []
    for left, right in zip(ids, ids[1:]):
        edges.extend([
            Relationship(left, right, RelationshipType.NEXT),
            Relationship(right, left, RelationshipType.PREVIOUS),
        ])
    return edges


def test_bidirectional_expansion_and_low_similarity_bridge():
    """Verify that bidirectional expansion and low similarity bridge."""
    scores = [("a", 0.9), ("seed", 0.95), ("bridge", 0.2), ("answer", 0.85)]
    retriever, plan, hits = setup_graph(scores, sequential_edges(["a", "seed", "bridge", "answer"]))
    evidence = retriever.retrieve("doc", plan, hits)
    assert set(evidence.visited_chunk_ids) == {"a", "seed", "bridge", "answer"}
    assert "bridge" not in evidence.included_chunk_ids
    assert "answer" in evidence.included_chunk_ids
    assert any(path.chunk_ids[-1] == "answer" for path in evidence.paths)


def test_multiple_seeds_are_deduplicated_and_cycles_terminate():
    """Verify that multiple seeds are deduplicated and cycles terminate."""
    scores = [("seed", 0.95), ("other", 0.91)]
    edges = [
        Relationship("seed", "other", RelationshipType.NEXT),
        Relationship("other", "seed", RelationshipType.NEXT),
    ]
    retriever, plan, hits = setup_graph(scores, edges)
    hits.append(SearchHit(hits[0].chunk, 0.8, 2))
    evidence = retriever.retrieve("doc", plan, hits)
    assert evidence.seed_chunk_ids == ("seed",)
    assert len(evidence.visited_chunk_ids) == 2


def test_all_global_hits_seed_expansion_by_default():
    """Verify that all global hits seed expansion by default."""
    scores = [(f"seed-{number}", 0.95 - number / 100) for number in range(5)]
    chunks = [
        Chunk(chunk_id, "doc", "sector", f"chunk {chunk_id}", number, token_count=1)
        for number, (chunk_id, _score) in enumerate(scores, start=1)
    ]
    retriever = AdaptiveGraphRetriever(
        chunks,
        (),
        similarity=lambda _query, chunk: dict(scores)[chunk.id],
    )
    plan = QueryPlan(
        "question",
        "factual",
        (QueryRequirement("main", "question", frozenset({"question"})),),
        False,
        False,
    )
    hits = tuple(
        SearchHit(chunk, score, rank)
        for rank, (chunk, (_chunk_id, score)) in enumerate(zip(chunks, scores), start=1)
    )

    evidence = retriever.retrieve("doc", plan, hits)

    assert evidence.seed_chunk_ids == tuple(chunk.id for chunk in chunks)
    assert set(evidence.included_chunk_ids) == {chunk.id for chunk in chunks}


def test_each_seed_gets_its_own_directional_expansion_budget():
    """Verify that each seed gets its own directional expansion budget."""
    seed_ids = [f"seed-{number}" for number in range(5)]
    neighbor_ids = ["neighbor-0a", "neighbor-0b", "neighbor-1", "neighbor-2", "neighbor-3", "neighbor-4"]
    chunks = [
        Chunk(chunk_id, "doc", "sector", chunk_id, number, token_count=1)
        for number, chunk_id in enumerate(seed_ids + neighbor_ids, start=1)
    ]
    scores = {chunk.id: 0.9 for chunk in chunks}
    edges = [
        Relationship("seed-0", "neighbor-0a", RelationshipType.NEXT),
        Relationship("seed-0", "neighbor-0b", RelationshipType.NEXT),
        *[
            Relationship(f"seed-{number}", f"neighbor-{number}", RelationshipType.NEXT)
            for number in range(1, 5)
        ],
    ]
    retriever = AdaptiveGraphRetriever(
        chunks,
        edges,
        lambda _query, chunk: scores[chunk.id],
        TraversalConfig(
            max_nodes=20,
            max_depth=1,
            max_backward=0,
            max_forward=1,
            include_threshold=0.6,
        ),
    )
    plan = QueryPlan(
        "question",
        "factual",
        (QueryRequirement("main", "question", frozenset({"question"})),),
        False,
        False,
    )
    hits = [
        SearchHit(next(chunk for chunk in chunks if chunk.id == seed_id), 0.9, rank)
        for rank, seed_id in enumerate(seed_ids, start=1)
    ]

    evidence = retriever.retrieve("doc", plan, hits)

    assert set(evidence.seed_chunk_ids) == set(seed_ids)
    assert set(evidence.visited_chunk_ids) >= {
        "neighbor-0a",
        "neighbor-1",
        "neighbor-2",
        "neighbor-3",
        "neighbor-4",
    }
    assert "neighbor-0b" not in evidence.visited_chunk_ids
    assert len(evidence.included_chunk_ids) == len(set(evidence.included_chunk_ids))


def test_rejected_edge_does_not_consume_directional_expansion_budget():
    """Verify that rejected edge does not consume directional expansion budget."""
    chunks = [
        Chunk("seed", "doc", "sector", "seed", 1, token_count=1),
        Chunk("rejected", "doc", "sector", "rejected", 2, token_count=1),
        Chunk("useful", "doc", "sector", "useful", 3, token_count=1),
    ]
    scores = {"seed": 0.9, "rejected": 0.1, "useful": 0.9}
    retriever = AdaptiveGraphRetriever(
        chunks,
        [
            Relationship("seed", "rejected", RelationshipType.NEXT),
            Relationship("seed", "useful", RelationshipType.NEXT),
        ],
        lambda _query, chunk: scores[chunk.id],
        TraversalConfig(
            max_nodes=10,
            max_depth=1,
            max_low_similarity_bridges=0,
            max_backward=0,
            max_forward=1,
            include_threshold=0.6,
        ),
    )
    plan = QueryPlan(
        "question",
        "factual",
        (QueryRequirement("main", "question", frozenset({"question"})),),
        False,
        False,
    )
    evidence = retriever.retrieve(
        "doc",
        plan,
        [SearchHit(chunks[0], 0.9, 1)],
    )

    assert "rejected" not in evidence.visited_chunk_ids
    assert "useful" in evidence.included_chunk_ids


def test_overlapping_seed_paths_traverse_bridges_but_deduplicate_evidence():
    """Verify that overlapping seed paths traverse bridges but deduplicate evidence."""
    chunks = [
        Chunk("seed-0", "doc", "sector", "seed 0", 1, token_count=1),
        Chunk("seed-1", "doc", "sector", "seed 1", 2, token_count=1),
        Chunk("bridge", "doc", "sector", "bridge", 3, token_count=1),
        Chunk("answer", "doc", "sector", "answer", 4, token_count=1),
    ]
    scores = {"seed-0": 0.99, "seed-1": 0.98, "bridge": 0.2, "answer": 0.9}
    retriever = AdaptiveGraphRetriever(
        chunks,
        [
            Relationship("seed-0", "bridge", RelationshipType.NEXT),
            Relationship("seed-1", "bridge", RelationshipType.NEXT),
            Relationship("bridge", "answer", RelationshipType.NEXT),
        ],
        lambda _query, chunk: scores[chunk.id],
        TraversalConfig(
            max_nodes=10,
            max_depth=3,
            max_low_similarity_bridges=1,
            max_backward=0,
            max_forward=2,
            include_threshold=0.6,
        ),
    )
    plan = QueryPlan(
        "question",
        "factual",
        (QueryRequirement("main", "question", frozenset({"question"})),),
        False,
        False,
    )
    hits = [
        SearchHit(chunks[index], scores[chunks[index].id], index + 1)
        for index in range(2)
    ]

    evidence = retriever.retrieve("doc", plan, hits)

    assert set(evidence.visited_chunk_ids) == {"seed-0", "seed-1", "bridge", "answer"}
    assert "bridge" not in evidence.included_chunk_ids
    assert evidence.included_chunk_ids.count("answer") == 1
    assert {
        path.chunk_ids
        for path in evidence.paths
        if path.chunk_ids[-1] == "answer"
    } == {
        ("seed-0", "bridge", "answer"),
        ("seed-1", "bridge", "answer"),
    }


def test_max_nodes_remains_a_document_wide_cap_after_seeds_are_processed():
    """Verify that max nodes remains a document wide cap after seeds are processed."""
    seed_ids = [f"seed-{number}" for number in range(5)]
    chunks = [
        Chunk(chunk_id, "doc", "sector", chunk_id, number, token_count=1)
        for number, chunk_id in enumerate(seed_ids, start=1)
    ]
    retriever = AdaptiveGraphRetriever(
        chunks,
        (),
        lambda _query, _chunk: 0.9,
        TraversalConfig(max_nodes=5, max_tokens_per_document=10),
    )
    plan = QueryPlan(
        "question",
        "factual",
        (QueryRequirement("main", "question", frozenset({"question"})),),
        False,
        False,
    )
    hits = [
        SearchHit(chunk, 0.9, rank)
        for rank, chunk in enumerate(chunks, start=1)
    ]

    evidence = retriever.retrieve("doc", plan, hits)

    assert evidence.visited_chunk_ids == tuple(seed_ids)
    assert evidence.included_chunk_ids == tuple(seed_ids)


def test_traversal_never_crosses_document_boundary():
    """Verify that traversal never crosses document boundary."""
    chunks = [
        Chunk("seed", "doc", "sector", "seed policy", 1, token_count=1),
        Chunk("foreign", "other-doc", "sector", "foreign policy", 2, token_count=1),
    ]
    retriever = AdaptiveGraphRetriever(
        chunks,
        [Relationship("seed", "foreign", RelationshipType.NEXT)],
        similarity=lambda _query, chunk: 0.95 if chunk.id == "seed" else 0.99,
    )
    plan = QueryPlan(
        "policy",
        "factual",
        (QueryRequirement("main", "policy", frozenset({"policy"})),),
        False,
        False,
    )
    hits = [SearchHit(chunks[0], 0.95, 1)]
    evidence = retriever.retrieve("doc", plan, hits)
    assert evidence.visited_chunk_ids == ("seed",)


def test_qualifies_relationship_is_followed_for_exception_queries():
    """Verify that qualifies relationship is followed for exception queries."""
    scores = [("seed", 0.9), ("exception", 0.8)]
    retriever, plan, hits = setup_graph(
        scores,
        [Relationship("seed", "exception", RelationshipType.QUALIFIES)],
    )
    evidence = retriever.retrieve("doc", plan, hits)
    assert "exception" in evidence.included_chunk_ids
    assert RelationshipType.QUALIFIES in evidence.relationship_types_followed


def test_low_similarity_qualifier_is_included_below_the_normal_threshold():
    """Verify that low similarity qualifier is included below the normal threshold."""
    scores = [("seed", 0.9), ("exception", 0.5)]
    retriever, plan, hits = setup_graph(
        scores,
        [Relationship("seed", "exception", RelationshipType.QUALIFIES)],
    )

    evidence = retriever.retrieve("doc", plan, hits)

    assert "exception" in evidence.visited_chunk_ids
    assert "exception" in evidence.included_chunk_ids


def test_qualifier_below_its_own_threshold_is_not_traversed():
    """Verify that qualifier below its own threshold is not traversed."""
    scores = [("seed", 0.9), ("exception", 0.4)]
    retriever, plan, hits = setup_graph(
        scores,
        [Relationship("seed", "exception", RelationshipType.QUALIFIES)],
    )

    evidence = retriever.retrieve("doc", plan, hits)

    assert "exception" not in evidence.visited_chunk_ids


def test_non_exception_queries_do_not_follow_qualifies_relationships():
    """Verify that non exception queries do not follow qualifies relationships."""
    scores = [("seed", 0.9), ("exception", 0.8)]
    retriever, plan, hits = setup_graph(
        scores,
        [Relationship("seed", "exception", RelationshipType.QUALIFIES)],
    )

    evidence = retriever.retrieve(
        "doc",
        replace(plan, asks_exceptions=False),
        hits,
    )

    assert "exception" not in evidence.visited_chunk_ids


def test_qualifies_edge_is_followed_from_a_sequentially_reached_chunk():
    """Verify that qualifies edge is followed from a sequentially reached chunk."""
    scores = [("seed", 0.9), ("rule", 0.9), ("exception", 0.5)]
    edges = sequential_edges(["seed", "rule"])
    edges.append(Relationship("rule", "exception", RelationshipType.QUALIFIES))
    retriever, plan, hits = setup_graph(scores, edges)

    evidence = retriever.retrieve("doc", plan, hits)

    assert "exception" in evidence.included_chunk_ids
    assert any(
        path.chunk_ids == ("seed", "rule", "exception")
        and path.relationship_types[-1] == RelationshipType.QUALIFIES
        for path in evidence.paths
    )


def test_node_limit_is_respected():
    """Verify that node limit is respected."""
    scores = [(f"c{number}", 0.9) for number in range(8)]
    retriever, plan, hits = setup_graph(scores, sequential_edges([item[0] for item in scores]))
    retriever.config = TraversalConfig(max_nodes=3, max_depth=10, max_forward=20, max_backward=20)
    evidence = retriever.retrieve("doc", plan, hits)
    assert len(evidence.visited_chunk_ids) <= 3


def test_similarity_is_computed_once_per_chunk_per_retrieval():
    """Verify that similarity is computed once per chunk per retrieval."""
    scores = [("seed", 0.95), ("bridge", 0.2), ("answer", 0.85)]
    retriever, plan, hits = setup_graph(
        scores,
        sequential_edges(["seed", "bridge", "answer"]),
    )
    original_similarity = retriever.similarity
    calls = {}

    def counted_similarity(query, chunk):
        """Count similarity calls before delegating to the test scorer."""
        calls[chunk.id] = calls.get(chunk.id, 0) + 1
        return original_similarity(query, chunk)

    retriever.similarity = counted_similarity
    retriever.retrieve("doc", plan, hits)

    assert calls
    assert max(calls.values()) == 1
