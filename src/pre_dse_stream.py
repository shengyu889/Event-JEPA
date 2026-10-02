import glob
import os
import shutil

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from pre_dse import Processor
from pre_dse_cli import build_parser, run_from_args
from utils import EventSlicer, accumulate_to_rgb


class StreamingProcessor(Processor):
    """
    Memory-bounded DSEC preprocessing.

    Difference from legacy Processor:
    - legacy loads the entire sequence event stream into RAM;
    - this implementation slices only one image interval at a time.

    The temporal boundary reproduces the legacy rule:
        current_event_timestamp > midpoint(current_image, next_image)
    and includes the first event timestamp that crosses that midpoint
    in the current event frame.
    """

    @staticmethod
    def _first_event_timestamp_after(
        slicer: EventSlicer,
        threshold: float,
        final_timestamp: int,
    ):
        # Legacy condition is strictly:
        # current_event_timestamp > threshold
        cursor = max(
            slicer.get_start_time_us(),
            int(np.floor(threshold)) + 1,
        )

        if cursor >= final_timestamp:
            return None

        # Search in bounded chunks. This does NOT load the whole sequence.
        search_span_us = 100_000  # 100 ms

        while cursor < final_timestamp:
            end = min(cursor + search_span_us, final_timestamp)

            if end <= cursor:
                return None

            events = slicer.get_events(cursor, end)

            if events is not None and events["t"].size > 0:
                return int(events["t"][0])

            cursor = end

        return None

    def _process_subfolder(self, subfolder):
        print(
            "---------------------------------- "
            "STREAM processing folder:",
            subfolder,
            flush=True,
        )

        event_dir = os.path.join(
            self.events_root,
            subfolder,
            "events",
            "left",
            "events.h5",
        )

        image_dir = os.path.join(
            self.image_root,
            subfolder,
            "images",
            "left",
            "rectified",
        )

        image_timestamp_dir = os.path.join(
            self.image_root,
            subfolder,
            "images",
            "timestamps.txt",
        )

        calib_dir = os.path.join(
            self.calib_root,
            subfolder,
            "calibration",
            "cam_to_cam.yaml",
        )

        save_root = os.path.join(
            self.image_root,
            subfolder,
            "images",
            "left",
        )

        warpped_dir = os.path.join(save_root, "warpped")
        vis_dir = os.path.join(save_root, "vis")
        eventImage_dir = os.path.join(save_root, "eventImage")

        label_dir = os.path.join(
            self.sementatic_root,
            subfolder,
            "11classes_renamed",
        )

        # ------------------------------------------------------------
        # Validate inputs
        # ------------------------------------------------------------
        for path in (
            event_dir,
            image_timestamp_dir,
            calib_dir,
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)

        image_names = sorted(
            glob.glob(os.path.join(image_dir, "*.png"))
        )

        image_timestamps = np.loadtxt(
            image_timestamp_dir,
            dtype=np.int64,
        )

        image_timestamps = np.atleast_1d(image_timestamps)

        if len(image_names) != len(image_timestamps):
            raise ValueError(
                f"{subfolder}: image/timestamp mismatch: "
                f"{len(image_names)} vs {len(image_timestamps)}"
            )

        if len(image_names) == 0:
            raise ValueError(
                f"{subfolder}: no rectified images found"
            )

        # ------------------------------------------------------------
        # Clean only temporary pair products.
        # eventToken is intentionally NOT deleted here.
        # ------------------------------------------------------------
        shutil.rmtree(warpped_dir, ignore_errors=True)
        shutil.rmtree(eventImage_dir, ignore_errors=True)
        shutil.rmtree(vis_dir, ignore_errors=True)

        os.makedirs(warpped_dir, exist_ok=True)
        os.makedirs(eventImage_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        print(
            "warpped images will be saved in:        "
            f"{warpped_dir}",
            flush=True,
        )
        print(
            "accumulated events will be saved in:    "
            f"{eventImage_dir}",
            flush=True,
        )

        # ------------------------------------------------------------
        # Camera / rectification mapping.
        # Unlike the old implementation, compute it only once.
        # ------------------------------------------------------------
        (
            H_homography,
            K_event,
            K_dist,
            dist_coeffs,
            resolution,
            Re,
        ) = self._get_camera(calib_dir)

        W, H = int(resolution[0]), int(resolution[1])

        mapping = cv2.initUndistortRectifyMap(
            K_dist,
            dist_coeffs,
            Re,
            K_event,
            (W, H),
            cv2.CV_32FC2,
        )[0]

        # ------------------------------------------------------------
        # Streaming event slicing
        # ------------------------------------------------------------
        with h5py.File(event_dir, "r") as event_file:
            slicer = EventSlicer(event_file)

            event_start = int(slicer.get_start_time_us())
            event_end = int(slicer.get_final_time_us())

            print(
                f"event start timestamp: {event_start}, "
                f"end timestamp: {event_end}, "
                f"duration: {(event_end-event_start)/1e6:.4f}s",
                flush=True,
            )
            print(
                f"image start timestamp: {int(image_timestamps[0])}, "
                f"end timestamp: {int(image_timestamps[-1])}",
                flush=True,
            )
            print(
                f"num of images: {len(image_names)}",
                flush=True,
            )

            window_start = event_start
            saved = 0

            for j in tqdm(
                range(len(image_names)),
                desc=f"stream {subfolder}",
            ):
                current_ts = int(image_timestamps[j])

                if j + 1 < len(image_timestamps):
                    next_ts = int(image_timestamps[j + 1])
                else:
                    # Match legacy implementation.
                    next_ts = current_ts

                midpoint = (current_ts + next_ts) / 2.0

                cutoff_ts = self._first_event_timestamp_after(
                    slicer,
                    midpoint,
                    event_end,
                )

                if cutoff_ts is None:
                    raise RuntimeError(
                        f"{subfolder}: no event found after "
                        f"midpoint={midpoint} for frame {j}"
                    )

                # get_events() is [start, end), therefore +1 includes
                # every event at cutoff_ts exactly like legacy code.
                window_end = cutoff_ts + 1

                if window_end > event_end:
                    window_end = event_end

                if window_end <= window_start:
                    raise RuntimeError(
                        f"{subfolder}: non-increasing event window "
                        f"{window_start} -> {window_end}"
                    )

                events = slicer.get_events(
                    window_start,
                    window_end,
                )

                if events is None:
                    raise RuntimeError(
                        f"{subfolder}: EventSlicer returned None for "
                        f"[{window_start}, {window_end})"
                    )

                # ----------------------------------------------------
                # Event frame
                # ----------------------------------------------------
                event_frame = accumulate_to_rgb(
                    events["x"],
                    events["y"],
                    events["p"],
                    (H, W),
                    pct=90,
                )

                event_frame = cv2.remap(
                    event_frame,
                    mapping,
                    None,
                    interpolation=cv2.INTER_CUBIC,
                )

                # ----------------------------------------------------
                # RGB warp retained for parity with legacy pipeline.
                # ----------------------------------------------------
                image = cv2.imread(image_names[j])

                if image is None:
                    raise RuntimeError(
                        f"failed to read image: {image_names[j]}"
                    )

                warped = cv2.warpPerspective(
                    image,
                    H_homography,
                    (W, H),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )

                event_path = os.path.join(
                    eventImage_dir,
                    f"{current_ts}.png",
                )

                warped_path = os.path.join(
                    warpped_dir,
                    f"{current_ts}.png",
                )

                if not cv2.imwrite(event_path, event_frame):
                    raise RuntimeError(
                        f"failed to write {event_path}"
                    )

                if not cv2.imwrite(warped_path, warped):
                    raise RuntimeError(
                        f"failed to write {warped_path}"
                    )

                # Optional semantic visualization.
                label_path = os.path.join(
                    label_dir,
                    f"{current_ts}.png",
                )

                if os.path.exists(label_path):
                    label = cv2.imread(label_path)

                    if label is not None:
                        denom = float(label.max() - label.min())

                        if denom > 0:
                            label_vis = (
                                (label - label.min()) / denom * 255
                            ).astype(np.uint8)

                            try:
                                vis_img = np.concatenate(
                                    [
                                        event_frame,
                                        warped,
                                        label_vis,
                                    ],
                                    axis=1,
                                )

                                cv2.imwrite(
                                    os.path.join(
                                        vis_dir,
                                        f"{current_ts}.png",
                                    ),
                                    vis_img,
                                )
                            except Exception:
                                pass

                # Next interval begins immediately after the timestamp
                # that was included in this frame.
                window_start = window_end
                saved += 1

        if saved != len(image_names):
            raise RuntimeError(
                f"{subfolder}: generated {saved} event frames "
                f"for {len(image_names)} images"
            )

        print(
            f"[STREAM OK] {subfolder}: "
            f"{saved}/{len(image_names)} pairs",
            flush=True,
        )

        return subfolder


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run_from_args(
            args,
            StreamingProcessor,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
