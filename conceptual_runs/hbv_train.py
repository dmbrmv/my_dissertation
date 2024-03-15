import multiprocessing as mp

from conceptual_runs.calibration.hbv_calibrator import gauges, hbv_single_core

gauges = gauges[:10]

if __name__ == "__main__":
    # Specify the number of processes to be used
    num_processes = 8  # You can adjust this value as needed

    # Create a multiprocessing pool with the specified number of processes
    pool = mp.Pool(processes=num_processes)

    # Map the calculation function to the list of numbers and run it in parallel
    results = pool.map(hbv_single_core, gauges)

    # Close the pool to free resources
    pool.close()
    pool.join()
