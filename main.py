import subprocess
import sys
import logging
import time
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline_step(script_path: str) -> None:
    logger.info(f"Starting pipeline step: {script_path}")
    start_time = time.time()

    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as exc:
        logger.critical(
            f"Pipeline step failed: {script_path} (return code: {exc.returncode}). Aborting pipeline."
        )
        sys.exit(1)

    duration_seconds = time.time() - start_time
    logger.info(
        f"Completed pipeline step: {script_path} in {duration_seconds:.2f} seconds"
    )


if __name__ == "__main__":
    total_start = time.time()

    scripts = [
        "src/models/baselines.py",
        "src/models/tree_models.py",
        "src/models/deep_learning.py",
        "src/models/ensembles.py",
    ]

    for script in scripts:
        if Path(script).exists():
            run_pipeline_step(script)
        else:
            logger.critical(f"Missing pipeline script: {script}. Aborting pipeline.")
            sys.exit(1)

    total_duration_minutes = (time.time() - total_start) / 60.0
    logger.info("=" * 72)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(
        f"All predictions are ready in the outputs folder (total runtime: {total_duration_minutes:.2f} minutes)."
    )
    logger.info("=" * 72)
