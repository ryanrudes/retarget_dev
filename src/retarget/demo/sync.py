"""Synchronization plans and execution helpers (string-keyed track names)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import networkx as nx

from retarget.demo.alignment import (
    EnergySignal,
    TimelineTransform,
    TrackAlignment,
    estimate_alignment_from_signals,
)
from retarget.demo.demo import Demonstration, DemonstrationView
from retarget.demo.resampling import ResampleMethod
from retarget.demo.tracks import Track

type SignalExtractor = Callable[[Track], EnergySignal]
"""A function that extracts a scalar signal from a track."""


@dataclass(frozen=True, slots=True)
class SyncEdge:
    """Pairwise synchronization request between two named tracks."""

    source: str
    reference: str
    source_signal: SignalExtractor
    reference_signal: SignalExtractor
    max_lag_seconds: float

    def __post_init__(self) -> None:
        if self.source == self.reference:
            raise ValueError("SyncEdge source and reference must be different")
        if self.max_lag_seconds < 0:
            raise ValueError("SyncEdge max_lag_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class SyncPlan:
    """Synchronization graph for a demonstration.

    ``reference`` is the root timeline for later composition/resampling. Edges
    may form any connected graph rooted at ``reference``; they are not required
    to be a star.
    """

    reference: str
    edges: tuple[SyncEdge, ...]

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
    def track_ids(self) -> frozenset[str]:
        """Track names required by this plan."""
        ids: set[str] = {self.reference}
        for edge in self.edges:
            ids.add(edge.source)
            ids.add(edge.reference)
        return frozenset(ids)

    @property
    def edge_ids(self) -> frozenset[tuple[str, str]]:
        """Directed edge ids in this plan."""
        return frozenset((edge.source, edge.reference) for edge in self.edges)


def estimate_sync(
    demonstration: Demonstration | DemonstrationView,
    plan: SyncPlan,
) -> tuple[TrackAlignment, ...]:
    """Estimate each pairwise alignment requested by a sync plan."""
    tracks = _tracks_of(demonstration)
    _validate_plan_tracks(tracks, plan)

    alignments: list[TrackAlignment] = []
    for edge in plan.edges:
        source_signal = edge.source_signal(tracks[edge.source])
        reference_signal = edge.reference_signal(tracks[edge.reference])
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


def estimate_sync_to_reference(
    demonstration: Demonstration | DemonstrationView,
    plan: SyncPlan,
) -> tuple[TrackAlignment, ...]:
    """Estimate root-reference alignments suitable for storing on a demo."""
    return compose_alignments_to_reference(
        reference=plan.reference,
        alignments=estimate_sync(demonstration, plan),
    )


def estimate_sync_and_resample_to_reference(
    demonstration: Demonstration | DemonstrationView,
    plan: SyncPlan,
    *,
    start: float,
    stop: float,
    method: ResampleMethod | str = ResampleMethod.NEAREST,
) -> DemonstrationView:
    """Estimate sync, slice the demo, and resample onto the plan reference.

    1. estimate pairwise sync edges from ``plan``;
    2. compose those alignments into direct transforms to ``plan.reference``;
    3. slice the demonstration to ``[start, stop)``;
    4. materialize that slice on the reference track timeline.

    ``method`` is the discrete sampling policy forwarded to every non-reference
    track during the final resample.
    """
    alignments = estimate_sync_to_reference(demonstration, plan)
    sliced = demonstration.slice_time(start, stop)
    aligned = DemonstrationView(
        source=sliced.source,
        tracks=dict(sliced.tracks),
        alignments=alignments,
    )
    return aligned.resample_to(plan.reference, method=method)


def compose_alignments_to_reference(
    *,
    reference: str,
    alignments: tuple[TrackAlignment, ...],
) -> tuple[TrackAlignment, ...]:
    """Compose pairwise alignments into root-reference transforms.

    Each returned alignment maps a non-reference track directly into
    ``reference`` time. Scores are not composed and are omitted.
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

    composed: list[TrackAlignment] = []
    for source in sorted(graph.nodes, key=str):
        if source == reference:
            continue
        composed.append(
            TrackAlignment(
                source=source,
                reference=reference,
                transform=_compose_path_to_reference(
                    graph, source=source, reference=reference
                ),
                score=None,
            )
        )
    return tuple(composed)


def _compose_path_to_reference(
    graph: nx.Graph,
    *,
    source: str,
    reference: str,
) -> TimelineTransform:
    path = nx.shortest_path(graph, source=source, target=reference)
    transform = TimelineTransform.identity()
    for start, stop in zip(path, path[1:]):
        transform = transform.then(_edge_transform(graph, start, stop))
    return transform


def _sync_graph(reference: str, edges: tuple[SyncEdge, ...]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(reference)
    for edge in edges:
        graph.add_edge(edge.source, edge.reference)
    return graph


def _alignment_graph(
    reference: str,
    alignments: tuple[TrackAlignment, ...],
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(reference)
    for alignment in alignments:
        if graph.has_edge(alignment.source, alignment.reference):
            raise ValueError(
                "Alignment graph contains duplicate undirected edge "
                f"between {alignment.source!r} and {alignment.reference!r}"
            )
        graph.add_edge(alignment.source, alignment.reference, alignment=alignment)
    return graph


def _edge_transform(graph: nx.Graph, source: str, reference: str) -> TimelineTransform:
    alignment: TrackAlignment = graph.edges[source, reference]["alignment"]
    if alignment.source == source and alignment.reference == reference:
        return alignment.transform
    if alignment.source == reference and alignment.reference == source:
        return alignment.transform.inverse()
    raise RuntimeError(
        "Alignment edge endpoints do not match graph endpoints; "
        "this indicates an internal sync graph construction bug."
    )


def _tracks_of(
    demonstration: Demonstration | DemonstrationView,
) -> Mapping[str, Track]:
    return cast(Mapping[str, Track], demonstration.tracks)


def _validate_plan_tracks(tracks: Mapping[str, Track], plan: SyncPlan) -> None:
    missing = plan.track_ids.difference(tracks)
    if missing:
        raise KeyError(f"SyncPlan references missing tracks: {sorted(missing)!r}")
