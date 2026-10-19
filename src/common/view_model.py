"""
Qt-free view logic of the GUI: which layers are visible and how far the playback has progressed.

`items` are Stage 2 waypoints or Stage 3 machine states (both have layer_index, x/y/z,
is_travel_move, feature_type; Stage 3 also has b/c).
"""
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


def layer_count(items: Sequence) -> int:
    return (max(i.layer_index for i in items) + 1) if len(items) else 0


def move_range(items: Sequence, layer_lo: int, layer_hi: int) -> Tuple[int, int]:
    """Index range [start, stop) of the items whose layer_index lies in [layer_lo, layer_hi]."""
    idx = [i for i, it in enumerate(items) if layer_lo <= it.layer_index <= layer_hi]
    return (idx[0], idx[-1] + 1) if idx else (0, 0)


def effective_layers(layer_lo: int, layer_hi: int, single_layer: bool, n_layers: int) -> Tuple[int, int]:
    """Clamps the requested range to 0..n_layers-1, keeps lo <= hi; `single_layer` shows only `layer_hi`."""
    if n_layers <= 0:
        return 0, -1
    hi = min(max(layer_hi, 0), n_layers - 1)
    lo = hi if single_layer else min(max(layer_lo, 0), hi)
    return lo, hi


@dataclass
class Playback:
    """Playback head over the moves of the visible range: `shown` of `total` moves are drawn."""
    total: int = 0
    shown: int = 0
    playing: bool = False
    speed: int = 1                    # multiplier of the base step

    def reset(self, total: int, show_all: bool = True) -> None:
        self.total = max(0, total)
        self.shown = self.total if show_all else 0
        self.playing = False

    @property
    def base_step(self) -> int:
        return max(1, self.total // 300)          # a full run takes about 300 ticks at 1x

    def advance(self) -> bool:
        """One timer tick. Returns False (and stops) when the end is reached."""
        if not self.playing:
            return False
        self.shown = min(self.total, self.shown + self.base_step * max(1, self.speed))
        if self.shown >= self.total:
            self.playing = False
            return False
        return True

    def toggle(self) -> bool:
        if not self.playing and self.shown >= self.total:
            self.shown = 0                        # restart from the beginning
        self.playing = not self.playing
        return self.playing

    def stop_index(self, start: int) -> int:
        """Exclusive global item index up to which moves are drawn."""
        return start + self.shown


def head_item(items: Sequence, start: int, shown: int) -> Optional[object]:
    """The item the nozzle is at (last drawn move of the visible range), or None."""
    if shown <= 0 or not len(items):
        return None
    return items[min(start + shown, len(items)) - 1]


def status_text(item, n_layers: int) -> str:
    if item is None:
        return "-"
    kind = "travel" if item.is_travel_move else item.feature_type
    s = f"Layer {item.layer_index + 1}/{n_layers} | {kind} | X{item.x:.2f} Y{item.y:.2f} Z{item.z:.2f}"
    if hasattr(item, "b") and hasattr(item, "c"):
        s += f" | B{item.b:.2f} C{item.c:.2f}"
    return s
