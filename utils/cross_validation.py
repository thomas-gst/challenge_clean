import subprocess
import itertools
import concurrent.futures
import queue
import threading
import time 

'''
Script used to launch cross_validation sweeps using
a few gpu's from the BR. The jobs are launched using a queue, 
ensuring no one gpu has to handle several jobs at once. 
'''

SERVERS = []


REMOTE_USER = ""
REMOTE_PROJECT_DIR = "" 

param_grid = {
    'heads': [1,2,3,4],
    'layers': [1,2,3],
    'intermediate_factor': [1,2,3],
    'learning_rate': [1e-2,1e-3],
    'weight_decay': [5e-2,1e-2,1e-3],
    'epochs': [70],
    'task_period': [5,6,7],
    'hidden_dim': [32,64,128],
}

keys, values = zip(*param_grid.items())
hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

server_queue = queue.Queue()
for server in SERVERS:
    server_queue.put(server)

results_list = []
results_lock = threading.Lock()

def parse_job_output(output_str):
    for line in output_str.splitlines():
        if line.startswith("BEST_VAL_LOSS:"):
            try:
                val = float(line.split(":")[1].strip())
            except ValueError:
                val = None
        if line.startswith("BEST_VAL_EPOCH:"):
            try:
                epoch = int(line.split(":")[1].strip())
            except ValueError:
                epoch = None
    return val, epoch
            
def run_job_on_server(hyperparams, job_id):
    server_address = None 
    try:
        server_address = server_queue.get(timeout=300)  
        print(f"[Job {job_id}] Acquired server: {server_address} for parameters: {hyperparams}") 
        cmd_args = ['uv', 'run', 'job.py']
        for key, value in hyperparams.items():
            cmd_args.append(f'--{key}')
            cmd_args.append(str(value))
            
        remote_script_command_str = " ".join(cmd_args)
        full_remote_shell_command = f"cd {REMOTE_PROJECT_DIR} && {remote_script_command_str}"
        
        ssh_command = [
            'ssh', 
            f'{REMOTE_USER}@{server_address}',
            full_remote_shell_command 
        ]
        
        print(f"[Job {job_id} Executing on {server_address}]: {' '.join(ssh_command)}")
        
        process = subprocess.run(ssh_command, capture_output=True, text=True, check=False)
        
        if process.returncode == 0:
            print(f"[Job {job_id}] Successfully executed on {server_address}.")
            val_loss, epoch = parse_job_output(process.stdout)
            if val_loss is not None:
                return {"params": hyperparams, "val_loss": val_loss, "epoch" : epoch, "server": server_address, "status": "success", "output": process.stdout}
            else:
                print(f"[Job {job_id}] Output parsing failed on {server_address}. Stdout:\\n{process.stdout}")
                return {"params": hyperparams, "val_loss": float('inf'), "server": server_address, "status": "parse_error", "output": process.stdout}
        else:
            print(f"[Job {job_id}] Failed on {server_address} with return code {process.returncode}.")
            error_message = process.stderr if process.stderr.strip() else "No stderr output."
            print(f"[Job {job_id}] Stderr: {error_message}")
            print(f"[Job {job_id}] Stdout: {process.stdout if process.stdout.strip() else 'No stdout output.'}")
            return {"params": hyperparams, "val_loss": float('inf'), "server": server_address, "status": "failed", "error": error_message, "output": process.stdout}
        
    except queue.Empty:
        print(f"[Job {job_id}] Timed out waiting for an available server.")
        return {"params": hyperparams, "val_loss": float('inf'), "server": "None", "status": "timeout_server_queue"}
    except Exception as e:
        print(f"[Job {job_id}] An unexpected error occurred for params {hyperparams} on server {server_address}: {e}")
        return {"params": hyperparams, "val_loss": float('inf'), "server": server_address, "status": "exception", "error": str(e)}
    finally:
        if server_address:
            server_queue.put(server_address)
            print(f"[Job {job_id}] Returned server: {server_address} to queue.")
            
def main_cross_validation():
    num_jobs = len(hyperparameter_combinations)
    print(f"Starting cross-validation with {num_jobs} jobs across {len(SERVERS)} servers.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SERVERS)) as executor:
        future_to_params = {}
        for i, params in enumerate(hyperparameter_combinations):
            job_id = f"job_{i+1:03d}"
            future = executor.submit(run_job_on_server, params, job_id)
            future_to_params[future] = params
            
        for future in concurrent.futures.as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                if result is None: 
                    print(f"[Job {job_id}] CRITICAL: run_job_on_server returned None for params {params}. This indicates an unhandled case.")
                    result = {"params": params, "val_loss": float('inf'), "epoch": float("inf"), "server": "Unknown", "status": "internal_none_return"}
                
                with results_lock:
                    results_list.append(result)
                
                status_val = result.get('status', 'unknown')
                loss_val = result.get('val_loss', 'N/A')
                server_val = result.get('server', 'N/A')
                epoch = result.get('epoch', 'N/A')
                print(f"[Job {job_id}] for params {params} completed. Status: {status_val}, Val Loss: {loss_val} at epoch :{epoch}, Server: {server_val}")

            except Exception as e:
                print(f"Job {job_id} (params {params}) generated an exception in future processing: {e}")
                with results_lock:
                    results_list.append({"params": params, "val_loss": float('inf'), "status": "future_exception", "error": str(e)})
                
    print("\\nCross-validation completed.\\n") 

    if not results_list:
        print("No results were collected from any jobs.")
        return

    
    successful_results = [
        r for r in results_list 
        if r is not None and \
           r.get('status') == 'success' and \
           isinstance(r.get('val_loss'), (int, float)) and \
           r.get('val_loss') != float('inf')
    ]
    
    print("\\n--- Full Results Log ---")
    for res_idx, res_item in enumerate(results_list):
        if res_item is not None:
            print(f"  Log {res_idx+1}: Params: {res_item.get('params', 'N/A')}, Loss: {res_item.get('val_loss', 'N/A')}, Server: {res_item.get('server', 'N/A')}, Status: {res_item.get('status', 'N/A')}")
            if res_item.get('status') not in ['success', 'parse_error'] and res_item.get('error'):
                 print(f"    Error: {res_item.get('error')}")
            if res_item.get('status') == 'parse_error' and res_item.get('output'):
                print(f"    Output (for parse_error):\\n{res_item.get('output')[:500]}...") 
        else:
            print(f"  Log {res_idx+1}: Result item was None.")


    if successful_results:
        best_result = min(successful_results, key=lambda x: x['val_loss'])
        print(f"\\n--- Best Result ---")
        print(f"Best validation loss: {best_result['val_loss']:.4f}")
        print(f"Achieved at epoch: {best_result['epoch']}")
        print(f"Parameters: {best_result['params']}")
        print(f"Achieved on server: {best_result['server']}")
    else:
        print("\\nNo successful job runs with valid validation loss found.")
        
if __name__ == "__main__":
    start_time = time.time()
    main_cross_validation()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")