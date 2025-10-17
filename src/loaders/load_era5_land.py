import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path("..").resolve()))

from src.meteo.era5_land_loader import download_era
from src.utils.logger import setup_logger

logger = setup_logger(name="ERA5Loader", log_file="logs/era_loader.log")


async def main() -> None:
    """Download ERA5 land data.

    Downloads meteorological data from ERA5-Land dataset for the specified
    date range and spatial extent using asynchronous processing.
    """
    try:
        await download_era(
            start_date="2007-01-01",
            last_date="2022-12-31",
            save_path="../../data/MeteoData/ParflowMeteo",
            meteo_variables=[
                "surface_solar_radiation_downwards",
                "surface_thermal_radiation_downwards",
                # "total_precipitation",
                # "2m_temperature",
                "2m_dewpoint_temperature",
                "surface_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            data_extent=[70.0, 20.0, 42.0, 45.0],  # North, West, South, East coordinates
            max_concurrent_downloads=6,
        )
        logger.info("ERA5-Land data download completed successfully")
    except Exception as e:
        logger.error(f"Failed to download ERA5-Land data: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
