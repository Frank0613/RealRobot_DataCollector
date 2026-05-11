"""
Interactive viewer for real_rgb / real_depth images stored in jetcobot.hdf5.

Usage:
    python3 tools/image_viewer.py                          # datasets/jetcobot.hdf5, all demos
    python3 tools/image_viewer.py --file datasets/my.hdf5  # custom file
    python3 tools/image_viewer.py --demo 2                 # jump to demo_2
    python3 tools/image_viewer.py --fps 30                 # playback speed

Keyboard:
    Space       play / pause
    Left / Right  prev / next frame
    D           toggle depth view
    Q / Esc     quit
"""

import argparse
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from matplotlib.animation import FuncAnimation


# ── colour-map for float32 depth ────────────────────────────────────────────

def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """Normalise a (H,W) float32 depth frame to a uint8 RGB colourmap."""
    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = float(valid.min()), float(valid.max())
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    coloured = plt.cm.plasma(norm)[:, :, :3]  # (H,W,3) float in [0,1]
    return (coloured * 255).astype(np.uint8)


# ── main viewer ─────────────────────────────────────────────────────────────

class DemoViewer:
    def __init__(self, filepath: str, demo_idx: int, fps: float):
        self.filepath = filepath
        self.fps = fps
        self.show_depth = False
        self._playing = False
        self._anim = None

        with h5py.File(filepath, "r") as f:
            root = f["data"]
            total = int(root.attrs.get("total", 0))
            if total == 0:
                sys.exit("[image_viewer] No demos found in file.")

            demo_idx = max(0, min(demo_idx, total - 1))
            demo = root[f"demo_{demo_idx}"]

            self.demo_idx  = demo_idx
            self.total     = total
            self.n_frames  = int(demo.attrs["num_samples"])
            self.arm_names = list(demo.attrs.get("arm_joint_names", []))

            self.rgb    = demo["real_rgb"][:]          # (T,H,W,3) uint8
            self.depth  = demo["real_depth"][:]        # (T,H,W) float32
            self.arm_q  = demo["arm_joint_pos"][:]     # (T,6)
            self.real_q = demo["real_joint_pos"][:]    # (T,7)
            self.stamps = demo["real_stamp"][:]        # (T,)

        self.frame_idx = 0
        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title(
            f"HDF5 Viewer — demo_{self.demo_idx}/{self.total - 1}  |  {self.filepath}"
        )

        gs = self.fig.add_gridspec(
            3, 2,
            height_ratios=[6, 1, 1],
            hspace=0.45, wspace=0.3,
        )

        self.ax_img    = self.fig.add_subplot(gs[0, 0])
        self.ax_joint  = self.fig.add_subplot(gs[0, 1])

        ax_slider_row  = self.fig.add_subplot(gs[1, :])
        ax_btn_row     = self.fig.add_subplot(gs[2, :])

        # Image panel
        self.ax_img.axis("off")
        self.im_handle = self.ax_img.imshow(
            self.rgb[0], interpolation="nearest"
        )
        self.title = self.ax_img.set_title("", fontsize=9)

        # Joint bar chart
        n_arm = len(self.arm_names)
        x = np.arange(n_arm)
        colours = ["steelblue"] * n_arm + ["coral"] * n_arm
        labels  = [f"sim\n{n}" for n in self.arm_names] + \
                  [f"real\n{n}" for n in self.arm_names]
        vals    = np.concatenate([self.arm_q[0], self.real_q[0, :n_arm]])
        self.bars = self.ax_joint.bar(
            np.arange(len(vals)), vals, color=colours
        )
        self.ax_joint.set_xticks(np.arange(len(vals)))
        self.ax_joint.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        self.ax_joint.set_ylabel("rad", fontsize=8)
        self.ax_joint.set_title("Joint positions (blue=sim, red=real)", fontsize=8)
        self.ax_joint.set_ylim(-np.pi, np.pi)
        self.ax_joint.axhline(0, color="grey", lw=0.5, ls="--")

        # Frame slider
        ax_slider_row.set_visible(False)
        slider_ax = self.fig.add_axes([0.12, 0.27, 0.78, 0.025])
        self.slider = mwidgets.Slider(
            slider_ax, "Frame", 0, self.n_frames - 1,
            valinit=0, valstep=1,
        )
        self.slider.on_changed(self._on_slider)

        # Buttons
        ax_btn_row.set_visible(False)
        btn_w, btn_h = 0.1, 0.045
        btn_y = 0.17

        ax_prev  = self.fig.add_axes([0.30, btn_y, btn_w, btn_h])
        ax_play  = self.fig.add_axes([0.43, btn_y, btn_w, btn_h])
        ax_next  = self.fig.add_axes([0.56, btn_y, btn_w, btn_h])
        ax_depth = self.fig.add_axes([0.69, btn_y, btn_w + 0.02, btn_h])

        self.btn_prev  = mwidgets.Button(ax_prev,  "◀ Prev")
        self.btn_play  = mwidgets.Button(ax_play,  "▶ Play")
        self.btn_next  = mwidgets.Button(ax_next,  "Next ▶")
        self.btn_depth = mwidgets.Button(ax_depth, "Depth: OFF")

        self.btn_prev.on_clicked(lambda _: self._step(-1))
        self.btn_play.on_clicked(lambda _: self._toggle_play())
        self.btn_next.on_clicked(lambda _: self._step(+1))
        self.btn_depth.on_clicked(lambda _: self._toggle_depth())

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._render_frame(0)

    # ── rendering ────────────────────────────────────────────────────────────

    def _render_frame(self, idx: int):
        self.frame_idx = int(np.clip(idx, 0, self.n_frames - 1))

        img = (
            depth_to_rgb(self.depth[self.frame_idx])
            if self.show_depth
            else self.rgb[self.frame_idx]
        )
        self.im_handle.set_data(img)

        label = "depth" if self.show_depth else "RGB"
        t     = self.stamps[self.frame_idx]
        self.title.set_text(
            f"demo_{self.demo_idx}  |  frame {self.frame_idx}/{self.n_frames-1}"
            f"  |  {label}  |  t={t:.3f}s"
        )

        # Update joint bars
        n_arm = len(self.arm_names)
        vals  = np.concatenate([
            self.arm_q[self.frame_idx],
            self.real_q[self.frame_idx, :n_arm],
        ])
        for bar, v in zip(self.bars, vals):
            bar.set_height(v)

        # Sync slider without re-triggering callback
        self.slider.eventson = False
        self.slider.set_val(self.frame_idx)
        self.slider.eventson = True

        self.fig.canvas.draw_idle()

    # ── controls ─────────────────────────────────────────────────────────────

    def _on_slider(self, val):
        if not self._playing:
            self._render_frame(int(val))

    def _step(self, delta: int):
        self._render_frame(self.frame_idx + delta)

    def _toggle_play(self):
        if self._playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        self._playing = True
        self.btn_play.label.set_text("⏸ Pause")
        interval_ms = max(1, int(1000 / self.fps))
        self._anim = FuncAnimation(
            self.fig, self._anim_step, interval=interval_ms,
            cache_frame_data=False,
        )
        self.fig.canvas.draw_idle()

    def _stop_play(self):
        self._playing = False
        self.btn_play.label.set_text("▶ Play")
        if self._anim is not None:
            self._anim.event_source.stop()
            self._anim = None

    def _anim_step(self, _frame):
        next_idx = self.frame_idx + 1
        if next_idx >= self.n_frames:
            self._stop_play()
            return
        self._render_frame(next_idx)

    def _toggle_depth(self):
        self.show_depth = not self.show_depth
        self.btn_depth.label.set_text(
            f"Depth: {'ON ' if self.show_depth else 'OFF'}"
        )
        self._render_frame(self.frame_idx)

    def _on_key(self, event):
        if event.key in ("q", "escape"):
            plt.close(self.fig)
        elif event.key == " ":
            self._toggle_play()
        elif event.key == "left":
            self._stop_play()
            self._step(-1)
        elif event.key == "right":
            self._stop_play()
            self._step(+1)
        elif event.key == "d":
            self._toggle_depth()

    def show(self):
        plt.show()


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="View images in a robot HDF5 dataset")
    parser.add_argument(
        "--file", default="datasets/jetcobot.hdf5",
        help="Path to the .hdf5 file (default: datasets/jetcobot.hdf5)",
    )
    parser.add_argument(
        "--demo", type=int, default=0,
        help="Demo index to view (default: 0)",
    )
    parser.add_argument(
        "--fps", type=float, default=15.0,
        help="Playback FPS (default: 15)",
    )
    args = parser.parse_args()

    viewer = DemoViewer(
        filepath=args.file,
        demo_idx=args.demo,
        fps=args.fps,
    )
    viewer.show()


if __name__ == "__main__":
    main()
