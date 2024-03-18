import multiprocessing as mp

from calibration.gr4j_calibrator import gauges, gr4j_single_core

# Define a shared flag using multiprocessing Manager
# manager = mp.Manager()
# exit_flag = manager.Value("b", False)

if __name__ == "__main__":
    # Specify the number of processes to be used
    num_processes = 8  # You can adjust this value as needed
    # Create a multiprocessing pool with the specified number of processes
    pool = mp.Pool(processes=num_processes)
    results = pool.map(gr4j_single_core, gauges)
    # try:
    #     # Map the calculation function to the list of numbers and run it in parallel
    #     results = pool.map(gr4j_single_core, gauges)
    #     if exit_flag.value:
    #         print("Exiting from main process due to exception.")
    # except Exception as e:
    #     print("An exception occurred in the main process:", e)
    # Close the pool to free resources
    pool.close()
    pool.join()
