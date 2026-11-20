#!/bin/bash -l
#SBATCH --job-name=sf-gpu-audit
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:25:00
#SBATCH --export=NONE
#SBATCH --output=/anvme/workspace/iwi5150h-slideflow_ci/logs/sf-gpu-%j.out
#SBATCH --error=/anvme/workspace/iwi5150h-slideflow_ci/logs/sf-gpu-%j.err

unset SLURM_EXPORT_ENV
set -euo pipefail

# -----------------------------------------------------------------------------
# NHR@FAU Alex NVIDIA A100/A40 GPU Verification Batch Job
# Mirrors helma_gpu_test.slurm on /anvme when Helma iwi5 H100 quota is full.
# -----------------------------------------------------------------------------

WORKSPACE_DIR="${WORKSPACE_DIR:-/anvme/workspace/iwi5150h-slideflow_ci}"
SIF_IMAGE="${SIF_IMAGE:-${WORKSPACE_DIR}/containers/slideflow-deps.sif}"
REPO_SRC="${REPO_SRC:-${WORKSPACE_DIR}/repo.git}"
TARGET_REF="${TARGET_REF:-HEAD}"
REPORT_TAG="${REPORT_TAG:-manual}"
REPORT_DIR="${WORKSPACE_DIR}/reports/${REPORT_TAG}-job${SLURM_JOB_ID}"

mkdir -p "${REPORT_DIR}" "${WORKSPACE_DIR}/hf_cache" "${WORKSPACE_DIR}/torch_cache"

export http_proxy="http://proxy.nhr.fau.de:80"
export https_proxy="http://proxy.nhr.fau.de:80"
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"
export NO_PROXY="localhost,127.0.0.1"

STAGE_REPO="${TMPDIR}/slideflow-src"
export GIT_LFS_SKIP_SMUDGE=1
GIT_NO_LFS=(git -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false)

"${GIT_NO_LFS[@]}" clone --no-checkout --shared "${REPO_SRC}" "${STAGE_REPO}"
cd "${STAGE_REPO}"
if "${GIT_NO_LFS[@]}" rev-parse --verify --quiet "origin/${TARGET_REF}" >/dev/null; then
    "${GIT_NO_LFS[@]}" checkout --detach "origin/${TARGET_REF}"
else
    "${GIT_NO_LFS[@]}" checkout --detach "${TARGET_REF}"
fi
COMMIT_SHA="$(git rev-parse --short HEAD)"
echo "Checked out : ${COMMIT_SHA} ($(git log -1 --pretty=%s)) on $(hostname)"

nvidia-smi

JUNIT_XML="${REPORT_DIR}/junit-gpu-${COMMIT_SHA}.xml"
MD_REPORT="${REPORT_DIR}/gpu-verification-${COMMIT_SHA}.md"

if [ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
    export APPTAINERENV_HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi
export APPTAINERENV_HF_HOME="${WORKSPACE_DIR}/hf_cache"
export APPTAINERENV_TORCH_HOME="${WORKSPACE_DIR}/torch_cache"
export APPTAINERENV_HTTP_PROXY="${http_proxy}"
export APPTAINERENV_HTTPS_PROXY="${https_proxy}"
export APPTAINERENV_http_proxy="${http_proxy}"
export APPTAINERENV_https_proxy="${https_proxy}"
export APPTAINERENV_PYTHONPATH="${STAGE_REPO}"
export APPTAINERENV_SF_BACKEND="torch"

set +e
apptainer exec --nv \
    --bind "${WORKSPACE_DIR}:${WORKSPACE_DIR}" \
    --bind "${TMPDIR}:${TMPDIR}" \
    "${SIF_IMAGE}" \
    pytest \
        "${STAGE_REPO}/slideflow/test/dataset_test.py" \
        "${STAGE_REPO}/slideflow/test/stats_test.py" \
        "${STAGE_REPO}/slideflow/test/norm_test.py" \
        "${STAGE_REPO}/scripts/ci/cpu_verification_suite.py" \
        "${STAGE_REPO}/scripts/hpc/gpu_verification_suite.py" \
        -v \
        --basetemp="${TMPDIR}/pytest_tmp" \
        --junitxml="${JUNIT_XML}"
PYTEST_EXIT=$?
set -e

apptainer exec --nv \
    --bind "${WORKSPACE_DIR}:${WORKSPACE_DIR}" \
    --bind "${TMPDIR}:${TMPDIR}" \
    "${SIF_IMAGE}" \
    python3 - "${JUNIT_XML}" "${MD_REPORT}" "${SLURM_JOB_ID}" "${COMMIT_SHA}" "${REPORT_TAG}" << 'EOF'
import datetime, os, sys, xml.etree.ElementTree as ET
import torch

xml_path, md_path, job_id, sha, tag = sys.argv[1:6]
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
cuda_ver = torch.version.cuda or "N/A"
torch_ver = torch.__version__

root = ET.parse(xml_path).getroot()
suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
failures = sum(int(s.attrib.get("failures", 0)) for s in suites)
errors = sum(int(s.attrib.get("errors", 0)) for s in suites)
skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)
duration = sum(float(s.attrib.get("time", 0.0)) for s in suites)
passed = tests - failures - errors - skipped
overall = "PASSED" if (failures == 0 and errors == 0) else "FAILED"

with open(md_path, "w", encoding="utf-8") as fh:
    fh.write(f"### NHR@FAU GPU Verification Report (`{tag}`)\n\n")
    fh.write(f"- **Status**: **{overall}**\n")
    fh.write(f"- **Commit**: `{sha}`\n")
    fh.write(f"- **Slurm Job ID**: `{job_id}` (`{os.uname().nodename}`)\n")
    fh.write(f"- **Hardware / Runtime**: `{gpu_name}` | PyTorch `{torch_ver}` (CUDA `{cuda_ver}`)\n")
    fh.write(f"- **Timestamp (UTC)**: `{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`\n\n")
    fh.write("| Total Tests | Passed | Failed | Errors | Skipped | Runtime |\n")
    fh.write("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
    fh.write(f"| {tests} | {passed} | {failures} | {errors} | {skipped} | {duration:.1f}s |\n")
EOF

cat "${MD_REPORT}"
exit "${PYTEST_EXIT}"
