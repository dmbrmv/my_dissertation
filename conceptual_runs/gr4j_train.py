import multiprocessing as mp

from calibration.gr4j_calibrator import gauges, gr4j_single_core

if __name__ == "__main__":
    # Specify the number of processes to be used
    num_processes = 8  # You can adjust this value as needed
    # Create a multiprocessing pool with the specified number of processes
    pool = mp.Pool(processes=num_processes)
    results = pool.map(gr4j_single_core, gauges)
    # Close the pool to free resources
    pool.close()
    pool.join()
