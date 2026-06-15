"""Synchronization plans and execution helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import networkx as nx

from retarget.core.enums import TrackId
from retarget.demo.alignment import (
    EnergySignal,
    TimelineTransform,
    TrackAlignment,
    estimate_alignment_from_signals,
)
from retarget.demo.demo import Demonstration, DemonstrationView
from retarget.demo.tracks import Track

type SignalExtractor = Callable[[Track], EnergySignal]
"""A function that extracts a scalar signal from a track."""


@dataclass(frozen=True, slots=True)
class SyncEdge[K: TrackId]:
    """Pairwise synchronization request between two tracks.

    A ``SyncEdge`` says: estimate an alignment from ``source`` into
    ``reference`` using scalar signals extracted from those tracks.
    """

    source: K
    reference: K
    source_signal: SignalExtractor
    reference_signal: SignalExtractor
    max_lag_seconds: float

    def __post_init__(self) -> None:
        if self.source == self.reference:
            raise ValueError("SyncEdge source and reference must be different")
        if self.max_lag_seconds < 0:
            raise ValueError("SyncEdge max_lag_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class SyncPlan[K: TrackId]:
    """Synchronization graph for a demonstration.

    ``reference`` is the root timeline for later composition/resampling. Edges
    may form any connected graph rooted at ``reference``; they are not required
    to be a star. Each edge estimates one pairwise alignment from ``source``
    into ``reference`` for that edge.

    The plan does not prescribe how aligned tracks are resampled; it only
    describes how to estimate pairwise timeline transforms.
    """

    reference: K
    edges: tuple[SyncEdge[K], ...]

    def __post_init__(self) -> None:
        if len(self.edges) == 0:
            raise ValueError("SyncPlan must contain at least one edge")

        edge_ids = [(edge.source, edge.reference) for edge in self.edges]
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("SyncPlan contains duplicate directed edges")

        undirected_edge_ids = [
            frozenset((edge.source, edge.reference)) for edge in self.edges
        ]
        if len(set(undirected_edge_ids)) != len(undirected_edge_ids):
            raise ValueError("SyncPlan contains duplicate undirected edges")

        if self.reference not in self.track_ids:
            raise ValueError("SyncPlan reference must appear in at least one edge")

        graph = _sync_graph(self.reference, self.edges)
        if not nx.is_connected(graph):
            reachable = frozenset(nx.node_connected_component(graph, self.reference))
            missing = self.track_ids.difference(reachable)
            raise ValueError(
                "SyncPlan graph must be connected to the plan reference; "
                f"unreachable tracks: {sorted(missing)!r}"
            )

    @property
    def track_ids(self) -> frozenset[K]:
        """Track ids required by this plan."""
        ids: set[K] = {self.reference}
        for edge in self.edges:
            ids.add(edge.source)
            ids.add(edge.reference)
        return frozenset(ids)

    @property
    def edge_ids(self) -> frozenset[tuple[K, K]]:
        """Directed edge ids in this plan."""
        return frozenset((edge.source, edge.reference) for edge in self.edges)


def estimate_sync[K: TrackId](
    demonstration: Demonstration[K],
    plan: SyncPlan[K],
) -> tuple[TrackAlignment[K], ...]:
    """Estimate each pairwise alignment requested by a sync plan."""
    _validate_plan_tracks(demonstration.tracks, plan)

    alignments: list[TrackAlignment[K]] = []
    for edge in plan.edges:
        source_track = demonstration.get_track(edge.source)
        reference_track = demonstration.get_track(edge.reference)

        source_signal = edge.source_signal(source_track)
        reference_signal = edge.reference_signal(reference_track)

        transform, score = estimate_alignment_from_signals(
            reference=reference_signal,
            source=source_signal,
            max_lag_seconds=edge.max_lag_seconds,
        )
        alignments.append(
            TrackAlignment(
                source=edge.source,
                reference=edge.reference,
                transform=transform,
                score=score,
            )
        )

    return tuple(alignments)


def estimate_sync_to_reference[K: TrackId](
    demonstration: Demonstration[K],
    plan: SyncPlan[K],
) -> tuple[TrackAlignment[K], ...]:
    """Estimate root-reference alignments suitable for storing on a demo.

    This first estimates each pairwise edge in ``plan`` and then composes the
    pairwise transforms through the sync graph so every returned alignment maps
    directly into ``plan.reference`` time.
    """
    return compose_alignments_to_reference(
        reference=plan.reference,
        alignments=estimate_sync(demonstration, plan),
    )


def estimate_sync_and_resample_to_reference[K: TrackId](
    demonstration: Demonstration[K],
    plan: SyncPlan[K],
    *,
    start: float,
    stop: float,
) -> DemonstrationView[K]:
    """Estimate sync, slice the demo, and resample onto the plan reference.

    This is a convenience wrapper for the common workflow:

    1. estimate pairwise sync edges from ``plan``;
    2. compose those alignments into direct transforms to ``plan.reference``;
    3. slice the demonstration to ``[start, stop)``;
    4. materialize that slice on the reference track timeline.

    The original ``Demonstration`` remains unchanged. The returned view carries
    the composed root-reference alignments used for resampling.
    """
    alignments = estimate_sync_to_reference(demonstration, plan)
    sliced = demonstration.slice_time(start, stop)
    aligned_view = DemonstrationView(
        source=sliced.source,
        tracks=sliced.tracks,
        alignments=alignments,
    )
    return aligned_view.resample_to(plan.reference)


def compose_alignments_to_reference[K: TrackId](
    *,
    reference: K,
    alignments: tuple[TrackAlignment[K], ...],
) -> tuple[TrackAlignment[K], ...]:
    """Compose pairwise alignments into root-reference transforms.

    Each returned alignment maps a non-reference track directly into
    ``reference`` time. Pairwise alignments may form any connected graph.
    Scores are not composed and are therefore omitted from the returned
    alignments.
    """
    if len(alignments) == 0:
        return ()

    graph = _alignment_graph(reference, alignments)
    if not nx.is_connected(graph):
        reachable = frozenset(nx.node_connected_component(graph, reference))
        missing = set(graph.nodes).difference(reachable)
        raise ValueError(
            "Alignment graph must be connected to the requested reference; "
            f"unreachable tracks: {sorted(missing)!r}"
        )

    composed: list[TrackAlignment[K]] = []
    for source in sorted(graph.nodes, key=str):
        if source == reference:
            continue
        composed.append(
            TrackAlignment(
                source=source,
                reference=reference,
                transform=_compose_path_to_reference(
                    graph,
                    source=source,
                    reference=reference,
                ),
                score=None,
            )
        )
    return tuple(composed)


def _compose_path_to_reference[K: TrackId](
    graph: nx.Graph,
    *,
    source: K,
    reference: K,
) -> TimelineTransform:
    """Compose transforms along the shortest path from source to reference."""
    path = nx.shortest_path(graph, source=source, target=reference)
    transform = TimelineTransform.identity()
    for start, stop in zip(path, path[1:]):
        transform = transform.then(_edge_transform(graph, start, stop))
    return transform


def _sync_graph[K: TrackId](
    reference: K,
    edges: tuple[SyncEdge[K], ...],
) -> nx.Graph:
    """Build the undirected sync topology for validation/path queries."""
    graph = nx.Graph()
    graph.add_node(reference)
    for edge in edges:
        graph.add_edge(edge.source, edge.reference)
    return graph


def _alignment_graph[K: TrackId](
    reference: K,
    alignments: tuple[TrackAlignment[K], ...],
) -> nx.Graph:
    """Build an undirected graph carrying directed pairwise alignments."""
    graph = nx.Graph()
    graph.add_node(reference)
    for alignment in alignments:
        if graph.has_edge(alignment.source, alignment.reference):
            raise ValueError(
                "Alignment graph contains duplicate undirected edge "
                f"between {alignment.source!r} and {alignment.reference!r}"
            )
        graph.add_edge(
            alignment.source,
            alignment.reference,
            alignment=alignment,
        )
    return graph


def _edge_transform[K: TrackId](
    graph: nx.Graph,
    source: K,
    reference: K,
) -> TimelineTransform:
    """Return the transform that maps ``source`` time to ``reference`` time."""
    alignment: TrackAlignment[K] = graph.edges[source, reference]["alignment"]
    if alignment.source == source and alignment.reference == reference:
        return alignment.transform
    if alignment.source == reference and alignment.reference == source:
        return alignment.transform.inverse()
    raise RuntimeError(
        "Alignment edge endpoints do not match graph endpoints; "
        "this indicates an internal sync graph construction bug."
    )


def _validate_plan_tracks[K: TrackId](
    tracks: Mapping[K, Track],
    plan: SyncPlan[K],
) -> None:
    missing = plan.track_ids.difference(tracks)
    if missing:
        raise KeyError(f"SyncPlan references missing tracks: {sorted(missing)!r}")