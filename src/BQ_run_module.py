"""
BQ_run_module.py
================
BisQue module entrypoint. Called by /module/PythonScriptWrapper.py once per
MEX execution, after the wrapper has fetched the input CXR image from the
BisQue server into the container.

Contract (BQ_module_generator convention):

    run_module(input_path_dict, output_folder_path) -> output_paths_dict

    * input_path_dict keys MUST match the input resource names declared in
      BISQUE.xml  ->  here: "CXR Image"
    * output_paths_dict keys MUST match the output resource names declared in
      BISQUE.xml  ->  here: "Generated Report"
"""

import json
import os

# NOTE: src.inference (torch/llava) is imported lazily inside run_module so the
# BISQUE_STUB integration path stays importable on a host without torch/GPU.

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")


def run_module(input_path_dict, output_folder_path):
    # Key MUST match the input name set with `bqmod inputs` / BISQUE.xml.
    input_img_path = input_path_dict["CXR Image"]

    # Integration-test stub: set BISQUE_STUB=1 to skip loading the 7B model and
    # emit a canned report instead. Lets the full BisQue round-trip (fetch ->
    # run -> upload) be tested on a CPU / low-RAM host (e.g. a MacBook) without
    # a GPU. Leave unset for real inference.
    if os.environ.get("BISQUE_STUB") == "1":
        report = (
            "[STUB REPORT] BisQue integration test — model not loaded.\n"
            f"Input image: {os.path.basename(input_img_path)}\n"
            "FINDINGS: (stubbed) The lungs are clear without focal consolidation. "
            "No pleural effusion or pneumothorax. Heart size is normal.\n"
            "IMPRESSION: (stubbed) No acute cardiopulmonary abnormality."
        )
    else:
        from .inference import load_pipeline, generate_report
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        pipeline = load_pipeline(cfg)
        report = generate_report(pipeline, input_img_path)

    stem = os.path.splitext(os.path.basename(input_img_path))[0]
    output_report_path = os.path.join(output_folder_path, f"{stem}_report.txt")
    with open(output_report_path, "w") as f:
        f.write(report)

    output_paths_dict = {}
    # Key MUST match the output name set with `bqmod outputs` / BISQUE.xml.
    output_paths_dict["Generated Report"] = output_report_path
    return output_paths_dict


if __name__ == "__main__":
    # Local smoke test (no BisQue):
    #   python -m src.BQ_run_module <path/to/cxr.png>
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.BQ_run_module <path/to/cxr.png>")
        sys.exit(1)

    input_path_dict = {"CXR Image": os.path.abspath(sys.argv[1])}
    output_folder_path = os.getcwd()
    out = run_module(input_path_dict, output_folder_path)

    report_path = out["Generated Report"]
    with open(report_path) as f:
        print("\n=== Generated Report ===")
        print(f.read())
    print(f"\n[saved to {report_path}]")
