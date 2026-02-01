# ------------------------------------------
# Imports
# ------------------------------------------
import sys
import argparse
import os
import subprocess

# ------------------------------------------
# Directories
# ------------------------------------------
module_dir = os.path.dirname(os.path.realpath(__file__))
root_dir_name = os.path.dirname(module_dir)

# ------------------------------------------
# Args
# ------------------------------------------
parser = argparse.ArgumentParser(
    description="This is script to run ingestion program for the competition"
)
parser.add_argument(
    "--codabench",
    help="True when running on Codabench",
    action="store_true",
)

# ------------------------------------------
# Main
# ------------------------------------------
if __name__ == "__main__":
    import sys
    
    # Force unbuffered output for immediate logging
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None
    
    # Write immediately to ensure logs are captured
    print("\n----------------------------------------------", flush=True)
    print("Ingestion Program started!", flush=True)
    print("----------------------------------------------\n\n", flush=True)
    print(f"[*] Python version: {sys.version}", flush=True)
    print(f"[*] Working directory: {os.getcwd()}", flush=True)
    print(f"[*] Script location: {__file__}", flush=True)
    print(f"[*] Command line args: {sys.argv}", flush=True)
    print(flush=True)

    try:
        from ingestion import Ingestion
        print("[*] Successfully imported Ingestion class", flush=True)
    except ImportError as e:
        print(f"[!] ERROR: Failed to import Ingestion: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    args = parser.parse_args()
    print(f"[*] Parsed arguments: codabench={args.codabench}", flush=True)

    if not args.codabench:
        # DO NOT CHANGE THESE PATHS UNLESS YOU CHANGE THE FOLDER NAMES IN THE BUNDLE
        input_dir = os.path.join(root_dir_name, "input_data")
        output_dir = os.path.join(root_dir_name, "sample_result_submission")
        program_dir = os.path.join(root_dir_name, "ingestion_program")
        submission_dir = os.path.join(root_dir_name, "sample_code_submission")
    else:
        # DO NOT CHANGE THESE PATHS. THESE ARE USED ON THE CODABENCH PLATFORM
        input_dir = "/app/input_data"
        output_dir = "/app/output"
        program_dir = "/app/program"
        submission_dir = "/app/ingested_program"

    sys.path.append(input_dir)
    sys.path.append(output_dir)
    sys.path.append(program_dir)
    sys.path.append(submission_dir)

    # Install requirements from submission (using professor's exact code)
    def install_requirements(submission_dir):
        """
        Checks for requirements.txt in the submission directory and installs them.
        """
        requirements_path = os.path.join(submission_dir, 'requirements.txt')
        
        if os.path.exists(requirements_path):
            print(f"Found requirements.txt at {requirements_path}. Installing dependencies...")
            
            # Construct the pip install command
            # sys.executable ensures we use the exact same Python interpreter running the ingestion
            cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_path]
            
            try:
                # Run the installation, capturing output helps with debugging logs
                subprocess.check_call(cmd)
                print("Dependencies installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"CRITICAL WARNING: Failed to install dependencies from requirements.txt. Error: {e}")
                # You can choose to sys.exit(1) here if you want to fail the submission immediately
        else:
            print("No requirements.txt found. Skipping dependency installation.")
    
    install_requirements(submission_dir)

    # Check if model.py exists before importing
    model_path = os.path.join(submission_dir, "model.py")
    print(f"[*] Looking for model.py at: {model_path}")
    if os.path.exists(model_path):
        print(f"[*] Found model.py, attempting import...")
    else:
        print(f"[!] ERROR: model.py not found at {model_path}")
        print(f"[!] Current working directory: {os.getcwd()}")
        print(f"[!] sys.path entries: {sys.path}")
        raise FileNotFoundError(f"model.py not found in submission directory: {submission_dir}")

    # Import model from submission dir using importlib (more reliable)
    print("[*] Importing model using importlib...", flush=True)
    import importlib.util
    import time
    
    try:
        # Load the module from file
        print("[*] Creating module spec...", flush=True)
        spec = importlib.util.spec_from_file_location("submission_model", model_path)
        if spec is None:
            raise ImportError(f"Could not load spec from {model_path}")
        print("[*] Module spec created", flush=True)
        
        print("[*] Creating module object...", flush=True)
        model_module = importlib.util.module_from_spec(spec)
        print("[*] Module object created", flush=True)
        
        # Execute the module (this is where imports happen)
        print("[*] Executing model.py (this may take a moment for TensorFlow to load)...", flush=True)
        start_time = time.time()
        
        try:
            spec.loader.exec_module(model_module)
            elapsed = time.time() - start_time
            print(f"[*] Module executed successfully in {elapsed:.2f} seconds", flush=True)
        except Exception as exec_error:
            elapsed = time.time() - start_time
            print(f"[!] ERROR during module execution (after {elapsed:.2f} seconds): {type(exec_error).__name__}: {exec_error}", flush=True)
            import traceback
            print("[!] Full traceback:", flush=True)
            traceback.print_exc()
            raise
        
        # Get the Model class
        print("[*] Checking for Model class...", flush=True)
        if not hasattr(model_module, 'Model'):
            raise AttributeError("model.py does not contain a 'Model' class")
        
        print("[*] Getting Model class...", flush=True)
        Model = model_module.Model
        print("[*] Successfully imported Model class", flush=True)
        
    except Exception as e:
        print(f"[!] ERROR: Failed to import model: {type(e).__name__}: {e}", flush=True)
        import traceback
        print("[!] Full traceback:", flush=True)
        traceback.print_exc()
        raise ImportError(f"Could not import Model from {model_path}: {e}")
    
    print("[*] Model class is ready to use", flush=True)
    print("[*] Model import completed successfully, proceeding with ingestion setup...", flush=True)

    # Wrap everything in try/except to catch any errors
    try:
        # Initialize Ingestions
        print("[*] Initializing Ingestion object...", flush=True)
        ingestion = Ingestion()
        print("[*] Ingestion object created", flush=True)

        # Start timer
        print("[*] Starting timer...", flush=True)
        ingestion.start_timer()
        print("[*] Timer started", flush=True)
        # Load train and test data
        print(f"[*] Loading train and test data from: {input_dir}", flush=True)
        ingestion.load_train_and_test_data(input_dir)
        print("[*] Data loading completed", flush=True)

        # initialize submission
        print("[*] Initializing submission model...", flush=True)
        ingestion.init_submission(Model)
        print("[*] Submission model initialized", flush=True)

        # fit submission
        print("[*] Starting model training...", flush=True)
        ingestion.fit_submission()
        print("[*] Model training completed", flush=True)

        # predict submission
        print("[*] Starting predictions...", flush=True)
        ingestion.predict_submission()
        print("[*] Predictions completed", flush=True)

        # compute result
        print("[*] Computing results...", flush=True)
        ingestion.compute_result()
        print("[*] Results computed", flush=True)

        # save result
        print(f"\n[*] Preparing to save results to: {output_dir}")
        print(f"[*] Output directory exists: {os.path.exists(output_dir)}")
        if not os.path.exists(output_dir):
            print(f"[*] Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        ingestion.save_result(output_dir)
        
        # Verify result file was created
        result_file = os.path.join(output_dir, "result.json")
        if os.path.exists(result_file):
            print(f"[✓] Verification: result.json exists at {result_file}")
            file_size = os.path.getsize(result_file)
            print(f"[✓] File size: {file_size} bytes")
        else:
            print(f"[!] ERROR: result.json was not created at {result_file}")
            print(f"[!] Listing files in output directory:")
            if os.path.exists(output_dir):
                for item in os.listdir(output_dir):
                    item_path = os.path.join(output_dir, item)
                    item_type = "DIR" if os.path.isdir(item_path) else "FILE"
                    item_size = os.path.getsize(item_path) if os.path.isfile(item_path) else 0
                    print(f"    - {item} ({item_type}, {item_size} bytes)")
            else:
                print(f"    Output directory does not exist!")
            raise FileNotFoundError(f"result.json was not created at {result_file}")

        # Stop timer
        ingestion.stop_timer()
        
    except Exception as e:
        # Stop timer even if there's an error
        try:
            ingestion.stop_timer()
        except:
            pass
        
        print(f"\n[!] ========================================")
        print(f"[!] ERROR during ingestion:")
        print(f"[!] ========================================")
        print(f"[!] {type(e).__name__}: {e}")
        import traceback
        print(f"[!] Traceback:")
        traceback.print_exc()
        print(f"[!] ========================================")
        
        # Try to save partial results if possible
        if hasattr(ingestion, 'ingestion_result') and ingestion.ingestion_result is not None:
            print(f"[!] Attempting to save partial results...")
            try:
                os.makedirs(output_dir, exist_ok=True)
                ingestion.save_result(output_dir)
                print(f"[!] Partial results saved")
            except Exception as save_error:
                print(f"[!] Failed to save partial results: {save_error}")
        
        # Re-raise the exception
        raise

    # Show duration
    print("\n------------------------------------")
    print(f"[✔] Total duration: {ingestion.get_duration()}")
    print("------------------------------------")

    # Save Duration
    ingestion.save_duration(output_dir)

    print("\n----------------------------------------------")
    print("[✔] Ingestion Program executed successfully!")
    print("----------------------------------------------\n\n")
