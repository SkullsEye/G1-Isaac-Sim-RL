import subprocess
import time
import csv
import argparse
from datetime import datetime
import threading

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not found. Install it with: pip install nvidia-ml-py")

def get_gpu_power(handle):
    """Returns power usage in Watts."""
    try:
        # NVML returns milliwatts, convert to Watts
        power_mW = pynvml.nvmlDeviceGetPowerUsage(handle)
        return power_mW / 1000.0
    except pynvml.NVMLError as error:
        return 0.0

def monitor_gpu(output_file, stop_event, interval=1.0):
    if not PYNVML_AVAILABLE:
        return

    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Monitor GPU 0

    print(f"🔋 Monitoring GPU 0 Power to {output_file}...")
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Time_Seconds", "Power_Watts"])
        
        start_time = time.time()
        while not stop_event.is_set():
            current_time = time.time()
            elapsed = current_time - start_time
            power = get_gpu_power(handle)
            
            writer.writerow([datetime.now().strftime("%H:%M:%S"), round(elapsed, 2), power])
            time.sleep(interval)

    pynvml.nvmlShutdown()
    print(f"✅ Logging complete. Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a command and track GPU energy.")
    parser.add_argument("--name", type=str, required=True, help="Name for the log file (e.g., run1)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="The command to run")
    args = parser.parse_args()

    if not args.command:
        print("Error: No command provided to run.")
        exit(1)

    # 1. Setup Logging
    log_filename = f"energy_log_{args.name}.csv"
    stop_event = threading.Event()
    
    # 2. Start Monitoring Thread
    monitor_thread = threading.Thread(target=monitor_gpu, args=(log_filename, stop_event))
    monitor_thread.start()

    # 3. Run the Training Command
    full_cmd = " ".join(args.command)
    print(f"🚀 Starting Training: {full_cmd}")
    
    try:
        # Run the training script and wait for it to finish
        subprocess.run(args.command, shell=False)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # 4. Stop Monitoring
        stop_event.set()
        monitor_thread.join()
