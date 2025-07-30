# scripts/run_test.ps1

Write-Output "============================================================"
Write-Output "==========      STARTING PIPELINE SMOKE TEST      =========="
Write-Output "============================================================"

# Set the environment variables for this terminal session
$env:PYTHONPATH = "."
# Set a special environment variable to suppress harmless warnings for cleaner output
$env:PYTHONWARNINGS="ignore"

# IMPORTANT: Make sure your conda environment is active
# > conda activate moevit

# --- Run the Test ---
# This is the single, clean command to run your entire pipeline test.
python train_test_pipeline.py

Write-Output ""
Write-Output "--- PIPELINE SMOKE TEST COMPLETED ---"