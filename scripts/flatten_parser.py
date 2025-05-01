import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class FlattenLogParser:
    """Parser for mris_flatten log files that tracks values over iterations."""

    def __init__(self, log_file=None):
        """Initialize the parser with an optional log file."""
        self.data = []
        self.params = {}
        self.final_error = None

        if log_file:
            self.parse_file(log_file)

    def parse_file(self, log_file):
        """Parse a mris_flatten log file."""
        with open(log_file, "r") as f:
            content = f.read()

        self._parse_content(content)
        return self.data

    def parse_string(self, content):
        """Parse log content from a string."""
        self._parse_content(content)
        return self.data

    def _parse_content(self, content):
        """Parse the log content."""
        # Parse initial parameters
        self._parse_parameters(content)

        # Parse iterations
        self._parse_iterations(content)

        # Parse final error
        self._parse_final_error(content)

    def _parse_parameters(self, content):
        """Parse the initial parameters."""
        # Match the first line containing parameters
        param_match = re.search(
            r"tol=([\d.e-]+), sigma=([\d.]+), host=(\w+), nav=(\d+), nbrs=(\d+), l_nlarea=([\d.]+), l_dist=([\d.]+)",
            content,
        )

        if param_match:
            self.params = {
                "tolerance": float(param_match.group(1)),
                "sigma": float(param_match.group(2)),
                "host": param_match.group(3),
                "nav": int(param_match.group(4)),
                "nbrs": int(param_match.group(5)),
                "l_nlarea": float(param_match.group(6)),
                "l_dist": float(param_match.group(7)),
            }

        # Match nlarea/dist ratios
        nlarea_matches = re.findall(r"nlarea/dist = ([\d.]+)", content)
        if nlarea_matches:
            self.params["nlarea_dist_ratios"] = [
                float(ratio) for ratio in nlarea_matches
            ]

    def _parse_iterations(self, content):
        """Parse all iteration lines in the log."""
        # Match iteration lines
        iter_pattern = r"(\d+): dt: ([\d.]+), sse: ([\d.]+) \(([\d.]+), ([\d.]+), ([\d.]+)\), neg: (\d+) \(%([^:]+):%([\d.]+)\), avgs: (\d+)"
        iterations = re.findall(iter_pattern, content)

        # Convert matches to dictionaries
        for iter_match in iterations:
            iter_data = {
                "iteration": int(iter_match[0]),
                "dt": float(iter_match[1]),
                "sse": float(iter_match[2]),
                "dist_error": float(iter_match[3]),
                "area_distortion": float(iter_match[4]),
                "angle_distortion": float(iter_match[5]),
                "neg_triangles": int(iter_match[6]),
                "neg_area_pct": float(iter_match[7]),
                "area_distortion_pct": float(iter_match[8]),
                "avgs": int(iter_match[9]),
            }
            self.data.append(iter_data)

    def _parse_final_error(self, content):
        """Parse the final distance error."""
        final_error_match = re.search(r"final distance error %([\d.]+)", content)
        if final_error_match:
            self.final_error = float(final_error_match.group(1))

    def to_dataframe(self):
        """Convert parsed data to a pandas DataFrame."""
        if not self.data:
            return pd.DataFrame()

        return pd.DataFrame(self.data)

    def plot_metrics(self, metrics=None, save_path=None):
        """
        Plot selected metrics over iterations.

        Parameters:
        - metrics: List of metrics to plot (defaults to ['sse', 'dist_error', 'neg_triangles'])
        - save_path: Path to save the plot (if None, plot will be displayed)
        """
        if not self.data:
            print("No data to plot.")
            return

        if metrics is None:
            metrics = ["sse", "dist_error", "neg_triangles"]

        df = self.to_dataframe()

        fig, axs = plt.subplots(
            len(metrics), 1, figsize=(12, 4 * len(metrics)), sharex=True
        )
        if len(metrics) == 1:
            axs = [axs]

        for i, metric in enumerate(metrics):
            if metric in df.columns:
                axs[i].plot(df["iteration"], df[metric])
                axs[i].set_ylabel(metric)
                axs[i].set_title(f"{metric} over iterations")
                axs[i].grid(True)
            else:
                print(f"Metric '{metric}' not found in data.")

        axs[-1].set_xlabel("Iteration")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def summarize(self):
        """Print a summary of the flattening process."""
        if not self.data:
            print("No data to summarize.")
            return

        print("=== mris_flatten Summary ===")
        print(f"Initial parameters: {self.params}")

        if self.data:
            first_iter = self.data[0]
            last_iter = self.data[-1]

            print("\nInitial values (iteration 0):")
            print(f"  SSE: {first_iter['sse']:.2f}")
            print(f"  Distance error: {first_iter['dist_error']:.3f}")
            print(
                f"  Negative triangles: {first_iter['neg_triangles']} ({first_iter['neg_area_pct']}%)"
            )

            print("\nFinal values (iteration {last_iter['iteration']}):")
            print(f"  SSE: {last_iter['sse']:.2f}")
            print(f"  Distance error: {last_iter['dist_error']:.3f}")
            print(
                f"  Negative triangles: {last_iter['neg_triangles']} ({last_iter['neg_area_pct']}%)"
            )

        if self.final_error is not None:
            print(f"\nFinal distance error: {self.final_error}%")

        print(f"\nTotal iterations: {len(self.data)}")


def main():
    """Main function to run the parser from command line."""
    parser = argparse.ArgumentParser(description="Parse mris_flatten log file.")
    parser.add_argument("log_file", help="Path to mris_flatten log file")
    parser.add_argument("--plot", action="store_true", help="Plot metrics")
    parser.add_argument("--output", "-o", help="Path to save parsed data as CSV")
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        default=["sse", "dist_error", "neg_triangles"],
        help="Metrics to plot (space separated)",
    )

    args = parser.parse_args()

    log_parser = FlattenLogParser(args.log_file)

    # Print summary
    log_parser.summarize()

    # Save to CSV if requested
    if args.output:
        df = log_parser.to_dataframe()
        df.to_csv(args.output, index=False)
        print(f"Data saved to {args.output}")

    # Plot if requested
    if args.plot:
        plot_path = args.output.replace(".csv", ".png") if args.output else None
        log_parser.plot_metrics(metrics=args.metrics, save_path=plot_path)


if __name__ == "__main__":
    main()
